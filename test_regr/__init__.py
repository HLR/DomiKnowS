import sys

if sys.platform == "win32":
    # On Windows, pyarrow's native library crashes the interpreter (access violation) when it first loads
    # after torch and scikit-learn, as happens when ``datasets`` is imported after the DomiKnowS graph.
    # Loading it first avoids that DLL conflict.
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pass
