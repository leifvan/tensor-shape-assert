import nox
import sys
v = sys.version.split(" ")[0]

# Reuse environments to speed things up locally (optional)
nox.options.reuse_venv = "yes"

@nox.session(python=v)
def numpy(session):
    session.install(".[dev]")
    session.install("numpy")
    session.env["TSA_TEST_LIBRARY"] = "numpy"
    session.run("pytest", "-q", external=True)

@nox.session(python=v)
def torch(session):
    session.install(".[dev]")
    session.install("numpy")
    session.install("torch")
    session.env["TSA_TEST_LIBRARY"] = "torch"
    session.run("pytest", "-q", external=True)

@nox.session(python=v)
def jax(session):
    session.install(".[dev]")
    session.install("jax")
    session.env["TSA_TEST_LIBRARY"] = "jax"
    session.env["JAX_ENABLE_X64"] = "true"
    session.run("pytest", "-q", external=True)

@nox.session(python=v)
def dask(session):
    session.install(".[dev]")
    session.install("dask[array]")
    session.env["TSA_TEST_LIBRARY"] = "dask"
    session.run("pytest", "-q", external=True)

@nox.session(python=v)
def ndonnx(session):
    session.install(".[dev]")
    session.install("onnxruntime")
    session.install("ndonnx")
    session.env["TSA_TEST_LIBRARY"] = "ndonnx"
    session.run("pytest", "-q", external=True)

@nox.session(python=v)
def sparse(session):
    session.install(".[dev]")
    session.install("sparse")
    session.env["TSA_TEST_LIBRARY"] = "sparse"
    session.run("pytest", "-q", external=True)

# @nox.session(python=v)
# def cupy(session):
#     session.install(".[dev]")
#     session.install("cupy")
#     session.env["TSA_TEST_LIBRARY"] = "cupy"
#     session.run("pytest", "-q", external=True)
