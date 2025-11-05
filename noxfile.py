import nox
import sys
# v = sys.version.split(" ")[0]

# Reuse environments to speed things up locally (optional)
nox.options.reuse_venv = "yes"

BACKENDS = {
    "numpy":   {"deps": ["numpy"], "env": {}},
    "torch":   {"deps": ["numpy", "torch"], "env": {}},
    "jax":     {"deps": ["jax"], "env": {"JAX_ENABLE_X64": "true"}},
    "dask":    {"deps": ["dask[array]"], "env": {}},
    "ndonnx":  {"deps": ["onnxruntime", "ndonnx"], "env": {}},
    "sparse":  {"deps": ["sparse"], "env": {}},
}

@nox.session()
@nox.parametrize("backend", sorted(BACKENDS.keys()))
def all_tests(session, backend: str):

    # get config
    cfg = BACKENDS[backend]

    # install dependencies and package
    session.install(".[dev]")
    if cfg["deps"]:
        session.install(*cfg["deps"])

    # set environment variables
    session.env["TSA_TEST_LIBRARY"] = backend
    for k, v in cfg["env"].items():
        session.env[k] = v

    # run tests
    session.run("pytest", "-q", external=True)

@nox.session()
def typecheck(session):
    """Run type checks using mypy."""
    session.install(".[dev]")
    session.install("numpy", "torch")
    session.install("mypy")
    session.run("mypy", "src")

