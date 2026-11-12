"""Identity and randomized numbering are both represented before test scoring."""
from ..protocol import digest


def identity_epoch(episode_id,count,epoch,seed=42):
    phase=int(digest([seed,episode_id,'anonymous-id-training-schedule-v1'])[:8],16)
    # With two objects, excluding identity would repeat the same swap forever.
    if count==2:return (epoch+phase)%2==0
    return epoch%4==phase%4
