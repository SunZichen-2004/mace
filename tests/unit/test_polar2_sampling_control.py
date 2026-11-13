import json
from mace.tools.polar2_sampling import update_sampling


def test_three_consecutive_validations_and_both_statistics(tmp_path,monkeypatch):
    p=tmp_path/'control.json'
    p.write_text(json.dumps({'weights':{'omol140m':.95,'spice_alpha':.01,'omol4m':0.,'bms25':.04}}))
    monkeypatch.setenv('POLAR2_SAMPLING_CONTROL',str(p))
    keys=['mae_alpha_field_A3','rmse_alpha_field_A3','mae_alpha_multipole_A3','rmse_alpha_multipole_A3']
    baseline=dict.fromkeys(keys,1.)
    update_sampling(baseline,None,'omol/spice_alpha')
    only_mae=dict(baseline,mae_alpha_field_A3=2.)
    for i in range(3):update_sampling(only_mae,i,'omol/spice_alpha')
    assert not json.loads(p.read_text()).get('events')
    for i in range(3,6):update_sampling(dict.fromkeys(keys,1.3),i,'omol/spice_alpha')
    s=json.loads(p.read_text());assert len(s['events'])==1
    assert abs(s['weights']['spice_alpha']-.02)<1e-10
    update_sampling(dict.fromkeys(keys,1.3),5,'omol/spice_alpha')
    assert len(json.loads(p.read_text())['events'])==1
    spin={'mae_spin_dipole':1.,'rmse_spin_dipole':1.}
    update_sampling(spin,None,'omol/omol4m')
    for i in range(3):update_sampling({k:1.5 for k in spin},i,'omol/omol4m')
    s=json.loads(p.read_text());assert abs(s['weights']['omol4m']-.03)<1e-10
    assert abs(sum(s['weights'].values())-1)<1e-10


def test_source_fraction_is_by_records_with_different_batch_sizes(monkeypatch):
    import numpy as np
    from torch.utils.data import ConcatDataset
    from mace.data.polar2_dataset import Polar2Dataset,Polar2BatchSampler
    leaves=[]
    for name,batch_size,p in [('a',128,.8),('b',64,.2)]:
        d=Polar2Dataset.__new__(Polar2Dataset)
        d.name=name;d.batch_size=batch_size;d.sample_probability=p
        d.ends=np.array([10000]);d.limit=None
        leaves.append(d)
    monkeypatch.delenv('POLAR2_SAMPLING_CONTROL',raising=False)
    monkeypatch.setenv('POLAR2_SAMPLE_BY_RECORD_COUNT','1')
    sampler=Polar2BatchSampler(ConcatDataset(leaves),steps=10000,seed=19)
    counts=[0,0]
    for ids in sampler:counts[int(ids[0]>=10000)]+=len(ids)
    assert abs(counts[0]/sum(counts)-.8)<.01
