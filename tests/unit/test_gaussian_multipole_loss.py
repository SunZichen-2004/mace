"""Online coefficient labels, atom normalization and exact validation aggregation."""
import numpy as np
import pytest
import torch
from ase import Atoms
from mace.data import AtomicData
from mace.data.utils import config_from_atoms,KeySpecification,update_keyspec_from_kwargs
from mace.tools.default_keys import DefaultKeys
from mace.tools.torch_geometric import DataLoader
from mace.tools.utils import AtomicNumberTable
from mace.modules.gaussian_loss import GaussianMultipoleLoss
from mace.tools.train import MACELoss


class ZeroBase(torch.nn.Module):
    def weighted_components(self,ref,pred,ddp=None):
        return {'energy':pred['energy'].sum()*0}


def graph(n,seed=1):
    rng=np.random.default_rng(seed)
    a=Atoms('H'*n,positions=rng.normal(size=(n,3)))
    a.info['REF_energy']=0.;a.info['total_charge']=0.;a.info['total_spin']=1.
    a.arrays['REF_forces']=np.zeros((n,3))
    a.arrays['mbis_charge_coefficients']=rng.normal(size=(n,4))
    a.arrays['mbis_spin_coefficients']=rng.normal(size=(n,4))
    keys=update_keyspec_from_kwargs(KeySpecification(),DefaultKeys.keydict())
    c=config_from_atoms(a,key_specification=keys)
    return AtomicData.from_config(c,z_table=AtomicNumberTable([1]),cutoff=6.,heads=['Default'])


def prediction(batch):
    n=len(batch.positions)
    offset=torch.arange(n*4,dtype=batch.positions.dtype).reshape(n,4)*.013
    return dict(energy=batch.energy.clone(),density_coefficients=(batch.mbis_charge_coefficients+offset).requires_grad_(),spin_density=(batch.mbis_spin_coefficients-offset*.3).requires_grad_())


@pytest.fixture(autouse=True)
def double():
    old=torch.get_default_dtype();torch.set_default_dtype(torch.double)
    yield
    torch.set_default_dtype(old)


def test_loader_and_force_style_atom_component_mean():
    b=next(iter(DataLoader([graph(4)],batch_size=1)));pred=prediction(b)
    loss=GaussianMultipoleLoss(ZeroBase())
    delta=loss.gaussian_residuals(b,pred)
    expected=sum(100*v.square().mean() for v in delta.values())
    torch.testing.assert_close(loss(b,pred),expected)
    expected.backward()
    assert pred['density_coefficients'].grad.abs().max()>0
    assert pred['spin_density'].grad.abs().max()>0
    duplicate=next(iter(DataLoader([graph(4),graph(4)],batch_size=2)))
    pp={k:(v.repeat(2,1) if k!='energy' else v.repeat(2)) for k,v in pred.items()}
    torch.testing.assert_close(loss(duplicate,pp),expected)


def test_uneven_batch_metric_is_global_atom_mean():
    loss=GaussianMultipoleLoss(ZeroBase());metrics=MACELoss(loss)
    refs=[];preds=[]
    for b in DataLoader([graph(2),graph(5),graph(3)],batch_size=2):
        pred=prediction(b);delta=loss.gaussian_residuals(b,pred)
        refs.append(delta);metrics.update(b,pred)
    value,aux=metrics.compute();expected=0.
    for key in ['charge','dipole','spin','spin_dipole']:
        v=torch.cat([d[key] for d in refs]).detach()
        assert aux['mae_gaussian_'+key]==pytest.approx(v.abs().mean().item())
        assert aux['rmse_gaussian_'+key]==pytest.approx(v.square().mean().sqrt().item())
        term=100*v.square().mean().item();expected+=term
        assert aux['loss_gaussian_'+key]==pytest.approx(term)
    assert value==pytest.approx(expected)


def test_missing_labels_and_pbc_fail():
    b=next(iter(DataLoader([graph(2)],batch_size=1)));p=prediction(b);loss=GaussianMultipoleLoss(ZeroBase())
    b.pbc[:]=True
    with pytest.raises(ValueError,match='nonperiodic'):loss(b,p)
    b.pbc[:]=False;b.mbis_spin_coefficients=None
    with pytest.raises(ValueError,match='MBIS'):loss(b,p)


def test_gaussian_option_default_off_and_custom_keys():
    from mace.tools.arg_parser import build_default_arg_parser
    parser=build_default_arg_parser()
    args=parser.parse_args(['--name=test','--train_file=test.xyz'])
    assert args.gaussian_weighted_multipole_training is False
    assert args.gaussian_multipole_weight==100
    keys=update_keyspec_from_kwargs(KeySpecification(),dict(mbis_charge_coefficients_key='Q',mbis_spin_coefficients_key='S'))
    assert keys.arrays_keys['mbis_charge_coefficients']=='Q'
    assert keys.arrays_keys['mbis_spin_coefficients']=='S'


def test_zero_config_weight_matches_force_mse_denominator():
    loss=GaussianMultipoleLoss(ZeroBase())
    b=next(iter(DataLoader([graph(2),graph(5)],batch_size=2)))
    b.weight[0]=0
    pred=prediction(b);delta=loss.gaussian_residuals(b,pred)
    expected=sum(100*(v.square()*b.weight[b.batch].view((-1,)+(1,)*(v.ndim-1))).mean() for v in delta.values())
    torch.testing.assert_close(loss(b,pred),expected)
    metric=MACELoss(loss);metric.update(b,pred);value,aux=metric.compute()
    assert value==pytest.approx(expected.item())
