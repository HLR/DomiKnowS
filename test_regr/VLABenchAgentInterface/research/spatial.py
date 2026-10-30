"""Conservative comparisons in an explicitly supplied coordinate frame."""
from math import sqrt, isfinite


def bounds(value):
    if value is None:
        return None
    if isinstance(value, tuple) and value and isinstance(value[0], tuple) and len(value[0]) == 2 and isinstance(value[0][0], str):
        value = dict(value)
    if isinstance(value, dict):
        low, high = value.get("lower"), value.get("upper")
    else:
        low = high = value
    if not isinstance(low, (list, tuple)) or not isinstance(high, (list, tuple)) or len(low) != 3 or len(high) != 3:
        raise ValueError("spatial evidence must be a 3D point or lower/upper bounds")
    if any(type(v) not in (int,float) or not isfinite(v) for v in (*low,*high)) or any(a>b for a,b in zip(low,high)):
        raise ValueError("spatial bounds must be finite and ordered")
    return tuple(low), tuple(high)


def less(a_low, a_high, b_low, b_high, *, inclusive=False):
    if a_high <= b_low if inclusive else a_high < b_low:
        return True
    if a_low > b_high if inclusive else a_low >= b_high:
        return False
    return None


def compare(requirement, ledger, *, planning=False):
    relation = requirement.get("relation")
    allowed = {"left", "right", "front", "behind", "above", "below", "inside", "disjoint", "near", "far", "on", "ordered"}
    if relation not in allowed:
        raise ValueError("unknown spatial relation")
    identifiers = requirement.get("entities", ()) if relation == "ordered" else (requirement.get("left"),requirement.get("right"))
    if len(identifiers)<2:
        return None
    rows = [ledger.get(i) for i in identifiers]
    frame = requirement.get("frame")
    if not frame or any(r is None or not (ledger.fresh(r.id) or planning and r.kind=="predicted" and r.episode==ledger.episode and r.value is not None) or r.frame != frame or r.units != "m" for r in rows):
        return None
    values = [bounds(r.value) for r in rows]
    if any(v is None for v in values):
        return None
    if relation == "ordered":
        axis = requirement.get("axis")
        if axis not in (0,1,2):
            return None
        if requirement.get("descending",False):values.reverse()
        checks = [less(a[0][axis],a[1][axis],b[0][axis],b[1][axis]) for a,b in zip(values,values[1:])]
        return False if False in checks else None if None in checks else True
    (al,ah),(bl,bh) = values
    directions = {"left":(0,False),"right":(0,True),"front":(1,False),"behind":(1,True),"below":(2,False),"above":(2,True)}
    if relation in directions:
        # Camera/world interpretations of these words must be resolved by the caller's frame contract.
        axis, reverse = directions[relation]
        axis = requirement.get("axis",axis)
        if axis not in (0,1,2):raise ValueError("spatial axis must be 0,1,2")
        return less(bl[axis],bh[axis],al[axis],ah[axis]) if reverse else less(al[axis],ah[axis],bl[axis],bh[axis])
    if relation in ("near","far"):
        threshold=requirement.get("threshold_m")
        if type(threshold) not in (int,float) or not isfinite(threshold) or threshold<0:return None
        minimum=sqrt(sum(max(0,a-d,c-b)**2 for a,b,c,d in zip(al,ah,bl,bh)))
        maximum=sqrt(sum(max(abs(a-d),abs(b-c))**2 for a,b,c,d in zip(al,ah,bl,bh)))
        near=True if maximum<=threshold else False if minimum>threshold else None
        return near if relation=="near" or near is None else not near
    # Box relations need actual volume bounds, not just uncertain object centres.
    if requirement.get("geometry") != "boxes":return None
    if relation == "inside":
        if ledger.value(requirement.get("interior")) is not True:return None
        return all(c<=a<=b<=d for a,b,c,d in zip(al,ah,bl,bh))
    if relation == "disjoint":return any(b<c or d<a for a,b,c,d in zip(al,ah,bl,bh))
    if relation == "on":
        contact=ledger.value(requirement.get("contact"))
        tolerance=requirement.get("tolerance_m")
        if type(contact) is not bool or type(tolerance) not in (int,float) or not isfinite(tolerance) or tolerance<0:return None
        return contact and abs(al[2]-bh[2])<=tolerance and all(ah[i]>=bl[i] and bh[i]>=al[i] for i in (0,1))
