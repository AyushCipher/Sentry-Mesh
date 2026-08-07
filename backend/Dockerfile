FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so a source change does not invalidate the wheel layer.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY sentrymesh ./sentrymesh
COPY backend ./backend
COPY eval ./eval

# Non-root: the API has no reason to hold write access to its own source.
RUN useradd --create-home --uid 10001 sentrymesh \
    && chown -R sentrymesh:sentrymesh /app
USER sentrymesh

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=4s --start-period=20s --retries=4 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
