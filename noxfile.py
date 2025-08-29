import nox

@nox.session(python="3.10")
def numpy(session):
    session.install(".")
    session.install("numpy")
    session.env["TSA_TEST_LIBRARY"] = "numpy"
    session.run("pytest", "-q", external=True)

@nox.session(python="3.10")
def torch(session):
    session.install(".")
    session.install("torch")
    session.env["TSA_TEST_LIBRARY"] = "torch"
    session.run("pytest", "-q", external=True)

# @nox.session(python="3.10")
# def jax(session):
#     session.install(".")
#     session.install("jax")
#     session.env["TSA_TEST_LIBRARY"] = "jax"
#     session.run("pytest", "-q", external=True)

# @nox.session(python="3.10")
# def dask(session):
#     session.install(".")
#     session.install("dask")
#     session.env["TSA_TEST_LIBRARY"] = "dask"
#     session.run("pytest", "-q", external=True)

# @nox.session(python="3.10")
# def ndonnx(session):
#     session.install(".")
#     session.install("ndonnx")
#     session.env["TSA_TEST_LIBRARY"] = "ndonnx"
#     session.run("pytest", "-q", external=True)

# @nox.session(python="3.10")
# def sparse(session):
#     session.install(".")
#     session.install("sparse")
#     session.env["TSA_TEST_LIBRARY"] = "sparse"
#     session.run("pytest", "-q", external=True)

# @nox.session(python="3.10")
# def cupy(session):
#     session.install(".")
#     session.install("cupy")
#     session.env["TSA_TEST_LIBRARY"] = "cupy"
#     session.run("pytest", "-q", external=True)

