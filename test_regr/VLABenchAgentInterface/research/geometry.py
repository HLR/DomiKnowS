"""Optional discrete configuration checks; never a continuous-safety certificate."""
from dataclasses import dataclass,asdict
import hashlib,json
import numpy as np


def path_digest(path):
    values=np.asarray(path,dtype=float)
    if values.ndim!=2 or not values.size or not np.isfinite(values).all():
        raise ValueError("joint path must be a nonempty finite matrix")
    return hashlib.sha256(json.dumps(values.tolist(),separators=(",",":")).encode()).hexdigest()


@dataclass(frozen=True)
class GeometryCertificate:
    status: str
    path_digest: str
    episode: str
    step: int
    revisions: tuple
    samples: int = 0
    minimum_clearance_m: float | None = None
    resolution: float = .02
    reason: str = ""
    continuous_safety: bool = False

    def as_dict(self):
        row=asdict(self);row["revisions"]=dict(self.revisions);return row


def check_sampled_path(path,ledger,clone,*,allowed_contacts=(),resolution=.02,clearance_m=.002,max_samples=10000):
    """clone() supplies an isolated context with set_configuration and signed_distances.

    Each distance is (geom_a, geom_b, signed_distance). The backend must include
    robot/held-object versus scene and self collision pairs. No live state is written.
    """
    digest=path_digest(path);values=np.asarray(path,dtype=float)
    if resolution<=0 or not np.isfinite(resolution) or clearance_m<0 or not np.isfinite(clearance_m):
        raise ValueError("invalid motion sampling/clearance")
    def certificate(status,samples=0,minimum=None,reason=""):
        return GeometryCertificate(status,digest,ledger.episode,ledger.step,ledger.revisions,samples,minimum,resolution,reason)
    if clone is None:return certificate("unavailable",reason="No isolated geometry/path backend")
    allowed={frozenset(pair) for pair in allowed_contacts}
    samples=[values[0]]
    for start,end in zip(values,values[1:]):
        count=max(1,int(np.ceil(np.max(np.abs(end-start))/resolution)))
        if len(samples)+count>max_samples:return certificate("unavailable",reason="Sampling budget exhausted")
        samples.extend(start+(end-start)*i/count for i in range(1,count+1))
    minimum=float("inf");checked=0
    try:
        with clone() as backend:
            if not getattr(backend,"complete_pairs",False):return certificate("unavailable",reason="Collision pair coverage is incomplete")
            if not getattr(backend,"held_geometry_complete",False):return certificate("unavailable",reason="Held-object geometry/attachment is unavailable")
            for configuration in samples:
                backend.set_configuration(configuration)
                distances=backend.signed_distances(clearance_m)
                for first,second,distance in distances:
                    if frozenset((first,second)) in allowed:continue
                    if not np.isfinite(distance):return certificate("unavailable",checked,reason="Nonfinite clearance")
                    minimum=min(minimum,float(distance))
                    if distance<clearance_m:return certificate("blocked",checked+1,minimum)
                checked+=1
    except (AttributeError,ValueError,TypeError,RuntimeError) as error:
        return certificate("unavailable",checked,reason=str(error))
    return certificate("clear",checked,None if minimum==float("inf") else minimum)


class MujocoPathClone:
    """Explicit joint path over an isolated MjData copy, with an audited geom-pair list."""
    def __init__(self,model,data,qpos_indices,pairs,*,held_object=None,held_body_update=None):
        self.model,self.data,self.indices,self.pairs=model,data,tuple(qpos_indices),tuple(pairs)
        self.held_body_update=held_body_update
        self.held_geometry_complete=held_object is False or held_object is True and callable(held_body_update)
        self.complete_pairs=False # Caller must supply a certified complete pair enumeration.

    def __enter__(self):
        import mujoco
        self.mujoco=mujoco
        if hasattr(mujoco,"mj_copyData"):
            self.copy=mujoco.MjData(self.model)
            mujoco.mj_copyData(self.copy,self.model,self.data)
        else:
            # MuJoCo 3.2 exposes independent MjData copying through __copy__.
            from copy import copy
            self.copy=copy(self.data)
        return self

    def __exit__(self,*_):self.copy=None

    def set_configuration(self,value):
        if len(value)!=len(self.indices):raise ValueError("joint path dimensions differ from qpos mapping")
        self.copy.qpos[list(self.indices)]=value
        self.mujoco.mj_forward(self.model,self.copy)
        if self.held_body_update is not None:
            self.held_body_update(self.model,self.copy)
            self.mujoco.mj_forward(self.model,self.copy)

    def signed_distances(self,margin):
        return [(a,b,self.mujoco.mj_geomDistance(self.model,self.copy,int(a),int(b),margin,None)) for a,b in self.pairs]
