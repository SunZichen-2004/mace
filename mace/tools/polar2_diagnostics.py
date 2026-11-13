"""Save failed batches and exact weights for independent numerical diagnostics."""
import json,os,time,logging
from pathlib import Path
import torch
import torch.distributed as dist


def save_failure(model,batch,captured,error):
    directory=os.environ.get('POLAR2_DIAGNOSTICS_DIR')
    if not directory:return
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    rank=dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    stem=f'failure_{time.time_ns()}_rank{rank}'
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [cpu(v) for v in value]
        return value
    report=dict(rank=rank,error=str(error),source=getattr(batch,'source_name',None),sample_ids=getattr(batch,'sample_id',None),parts={k:float(v.detach()) for k,v in captured.get('parts',{}).items()})
    payload=dict(metadata=report,batch=cpu(batch.to_dict()))
    if rank==0:payload['model']=cpu(model.state_dict())
    temporary=root/(stem+'.tmp');torch.save(payload,temporary);temporary.replace(root/(stem+'.pt'))
    (root/(stem+'.json')).write_text(json.dumps(report,indent=2))
    logging.error('Saved POLAR-2 failure diagnostic: %s',root/(stem+'.pt'))
