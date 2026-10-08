# The Sandbox image: a fixed Python with the analysis libraries, nothing else.
# perry/sandbox.py tags it by this file's hash, so editing it triggers a rebuild.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
        numpy==2.1.3 \
        pandas==2.2.3 \
        scipy==1.14.1 \
        matplotlib==3.9.2 \
    && useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin sandbox

ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER sandbox
WORKDIR /work
# The script arrives on stdin.
ENTRYPOINT ["python", "-"]
