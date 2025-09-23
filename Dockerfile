# hadolint global ignore=DL3008
# kics-scan disable=965a08d7-ef86-4f14-8792-4a3b2098937e,451d79dc-0588-476a-ad03-3c7f0320abb3
FROM ghcr.io/astral-sh/uv:0.8.21@sha256:ca74b4b463d7dfc1176cbe82a02b6e143fd03a144dcb1a87c3c3e81ac16c6f6d AS uv

##
# base
##
FROM debian:stable-slim@sha256:0c80836db6d5bd1f89ed3c1300bbd888fc8a415130a52fd86e501ce38471fe5b AS base

# set up user
ARG USER=user
ARG UID=1000
RUN useradd --create-home --shell /bin/false --uid ${UID} ${USER}

# set up environment
ARG APP_HOME=/work/app
ARG VIRTUAL_ENV=${APP_HOME}/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    UV_LOCKED=1 \
    UV_NO_SYNC=1 \
    UV_PYTHON_DOWNLOADS=manual \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=${VIRTUAL_ENV}

WORKDIR ${APP_HOME}

##
# dev
##
FROM base AS dev

ARG DEBIAN_FRONTEND=noninteractive
COPY <<-EOF /etc/apt/apt.conf.d/99-disable-recommends
APT::Install-Recommends "false";
APT::Install-Suggests "false";
APT::AutoRemove::RecommendsImportant "false";
APT::AutoRemove::SuggestsImportant "false";
EOF

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG PYTHONDONTWRITEBYTECODE=1
ARG UV_NO_CACHE=1

# set up python
COPY --from=uv /uv /uvx /bin/
COPY .python-version pyproject.toml uv.lock ./
RUN uv python install && \
    uv sync --no-default-groups --no-install-project && \
    chown -R "${USER}:${USER}" "${VIRTUAL_ENV}" && \
    chown -R "${USER}:${USER}" "${APP_HOME}" && \
    uv pip list

# set up project
COPY xfmr_rec xfmr_rec
RUN uv sync --no-default-groups

USER ${USER}
HEALTHCHECK CMD [ uv, run, lightning, fit, --print_config ]
