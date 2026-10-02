"""Bundle only repro code, handoff and the verified final report (no library)."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib

here=Path(__file__).resolve().parent
destination=here.parent.parent/'DomiKnowS_ILP_repro_20260928.zip'
names=['conftest.py','run_suite.py','test_components.py','test_symbolic_bindings.py',
       'test_native.py','test_performance.py','HANDOFF.txt','README.md','build_bundle.py']
names += ['results/20260928T200637Z.'+extension for extension in ('json','xml','log')]
with ZipFile(destination,'w',ZIP_DEFLATED) as archive:
    for name in names:
        archive.write(here/name, 'ilp_repro_20260928/'+name)
with ZipFile(destination) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist())==len(names)
print(destination)
print('SHA256',hashlib.sha256(destination.read_bytes()).hexdigest())
print('FILES',len(names),'BYTES',destination.stat().st_size)
