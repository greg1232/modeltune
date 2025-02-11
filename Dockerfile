
###############################################################################
# CPU BASE IMAGE
FROM python:3.10 AS cpu

ARG INSTALL_ROOT=/app

# RUN pip install virtualenv==20.28.1

ENV VIRTUAL_ENV=/app/.poetry-venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV
RUN . $VIRTUAL_ENV/bin/activate

COPY ./modelgauge ${INSTALL_ROOT}/modelgauge

WORKDIR ${INSTALL_ROOT}/modelgauge

RUN curl -sSL https://install.python-poetry.org | python -

ENV PATH="/root/.local/bin:$PATH"

RUN poetry install
###############################################################################


