"""Isolated job tests: no credentials, cloud objects or datasets are touched."""
import ast
import json
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request


def test_jobs():
    tree = ast.parse(Path(__file__).with_name('reannotation_app.py').read_text(encoding='utf-8'))
    names = {'_run_bulk_registration', 'api_oss_bulk_status', 'api_oss_bulk_register'}
    app = Flask(__name__)
    gate = threading.Event()

    class Storage:
        enabled = True
        fail = False

        def put_bytes(self, *args):
            assert gate.wait(5)
            if self.fail:
                raise TimeoutError('simulated OSS timeout')

    storage = Storage()
    ns = dict(globals(), app=app, R2=storage, _bulk_job={}, _bulk_job_lock=threading.Lock(),
              _valid_direct_upload_id=lambda value: value.startswith('direct_'),
              _safe_upload_relative_path=lambda value: value,
              _complete_direct_upload=lambda value: jsonify(success=True, total=3))
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), '<jobs>', 'exec'), ns)
    client = app.test_client()
    payload = dict(upload_id='direct_' + 'a' * 32, files=['images/a.jpg'])
    started = time.monotonic()
    response = client.post('/api/oss_bulk/register', json=payload)
    assert response.status_code == 202 and time.monotonic() - started < 1
    job_id = response.json['job_id']
    assert client.post('/api/oss_bulk/register', json=payload).json['job_id'] == job_id
    assert client.post('/api/oss_bulk/register', json=dict(payload, upload_id='direct_b')).status_code == 409
    assert client.get('/api/oss_bulk/status/' + job_id).json['state'] == 'running'
    gate.set()

    def finished(identifier):
        for _ in range(100):
            status = client.get('/api/oss_bulk/status/' + identifier).json
            if status['state'] != 'running':
                return status
            time.sleep(.01)
        raise AssertionError('job did not finish')

    assert finished(job_id)['state'] == 'completed'
    storage.fail = True
    retry_id = client.post('/api/oss_bulk/register', json=payload).json['job_id']
    failed = finished(retry_id)
    assert failed['state'] == 'failed' and 'TimeoutError' in failed['error']
    assert client.get('/api/oss_bulk/status/missing').status_code == 404
    template = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HTML_TEMPLATE' for t in n.targets))
    script = template.split('<script>', 1)[1].split('</script>', 1)[0]
    subprocess.run(['node', '--check'], input=script, text=True, check=True, encoding='utf-8')


if __name__ == '__main__':
    test_jobs()
    print('Job success, timeout, deduplication, conflict, missing task and rendered JS checks passed.')
