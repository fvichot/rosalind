# -*- coding: utf8 -*-
import os
import re
import requests
import sys
import time
from termcolor import colored

RC_FILE = os.path.join(os.environ['HOME'], ".rosalindrc")
BASE_URL = 'http://rosalind.info'
CSRF_TOKEN_RE = re.compile(r"type='hidden' name='csrfmiddlewaretoken' value='([^']+)'")
SUCCESS_TEXT = r'<span class="label label-success">Congratulations</span>'


def print_step(msg):
    print colored(msg, attrs=['bold'])


def fail(msg):
    print colored(msg, 'red', attrs=['bold'])
    sys.exit(-1)


def _get_csrf_token(session, url):
    r = session.get(url)
    if r.status_code != 200:
        fail("Could not load {} !".format(url))
    m = CSRF_TOKEN_RE.search(r.text)
    if m is None:
        fail("Could not get the CSRF token!")
    return m.group(1)


def _login():
    user, password = (None, None)
    with open(RC_FILE, 'r') as f:
        user, password = map(lambda x: x.strip(), f.readlines())
    print_step("Logging in...")
    session = requests.Session()
    csrf_token = _get_csrf_token(session, BASE_URL + '/accounts/login/')
    data = {"username": user, "password": password, "csrfmiddlewaretoken": csrf_token}
    r = session.post(BASE_URL + '/accounts/login/', data=data)
    if r.status_code != 200:
        fail("Logging in failed!")
    return session


def _get_dataset(session, problem):
    print_step("Retrieving dataset for {}...".format(problem))
    r = session.get(BASE_URL + '/problems/{}/dataset/'.format(problem))
    if r.status_code != 200:
        fail("Failed to retrieve dataset!")
    return r.text


def _send_solution(session, problem, solution):
    csrf_token = _get_csrf_token(session, BASE_URL + '/problems/{}/'.format(problem))
    data = {"output_text": solution, "csrfmiddlewaretoken": csrf_token}
    r = session.post(BASE_URL + '/problems/{}/'.format(problem), data=data)
    if r.status_code != 200:
        fail("Posting solution failed!")
    r = session.get(BASE_URL + '/problems/{}/'.format(problem))
    if r.status_code != 200:
        fail("Could not check for success for {} !".format(problem))
    m = r.text.find(SUCCESS_TEXT)
    return (m != -1)


def solve(problem, func, silent=False):
    session = _login()
    dataset = _get_dataset(session, problem)
    if not silent:
        print dataset
    start_time = time.time()
    solution = func(dataset)
    duration = time.time() - start_time
    if not silent:
        print solution
    print_step("Generated solution in {:.4f}s".format(duration))
    success = _send_solution(session, problem, solution)
    if success:
        print colored("Success!", 'green', attrs=['bold'])
    else:
        fail("Wrong result!")


if __name__ == '__main__':
    solve(sys.argv[1], lambda x: x)
