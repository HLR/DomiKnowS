"""Identify the standard NF4/BF16 backend without changing its arithmetic."""
POLICY='bitsandbytes-default-nf4-bf16-v1'


def default_policy():
    import bitsandbytes as bnb
    if not bnb.matmul_4bit.__module__.startswith('bitsandbytes.'):
        raise ValueError('unexpected replacement of the standard quantized backend')
    return POLICY
