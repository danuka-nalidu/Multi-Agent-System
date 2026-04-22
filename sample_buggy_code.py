"""
sample_buggy_code.py
====================
Intentionally flawed Python file used to demonstrate the CTSE MAS pipeline.
Contains examples of code quality bugs AND security vulnerabilities.

DO NOT USE THIS CODE IN PRODUCTION.
"""

import os
import random
import pickle
import hashlib
import sqlite3

# ── Hardcoded credentials (Security: HardcodedSecret, OWASP A02) ─────────────
password = "supersecret123"
api_key = "sk-prod-abc123456789"
DB_HOST = "localhost"

DEBUG = True  # Security: DebugModeEnabled, OWASP A05

# ── Mutable default argument (Bug: MutableDefaultArgument) ───────────────────
def add_user_to_list(user, user_list=[]):
    user_list.append(user)
    return user_list

# ── Bare except (Bug: BareExcept) ─────────────────────────────────────────────
def read_config(path):
    try:
        with open(path) as f:
            return f.read()
    except:
        return None

# ── Division by zero (Bug: ZeroDivision) ──────────────────────────────────────
def bad_average():
    total = 100
    count = 0
    return total / 0

# ── Weak crypto (Security: WeakCryptography, OWASP A02) ──────────────────────
def hash_password(pw):
    return hashlib.MD5(pw.encode()).hexdigest()

# ── SQL Injection (Security: SQLInjection, OWASP A03) ─────────────────────────
def get_user(username):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    # DANGER: string concatenation with user input
    user = username
    query = "SELECT * FROM users WHERE name = '" + user + "'"  # noqa
    cur.execute(query)
    return cur.fetchall()

# ── Insecure deserialization (Security: InsecureDeserialization, OWASP A08) ───
def load_session(session_bytes):
    return pickle.loads(session_bytes)

# ── Insecure random (Security: InsecureRandom, OWASP A02) ────────────────────
def generate_token():
    return str(random.randint(100000, 999999))

# ── Code injection (Security: CodeInjection, OWASP A03) ──────────────────────
def run_command(user_input):
    exec(user_input)

# ── Assert in production (Bug: AssertInProduction) ────────────────────────────
def process_order(amount):
    assert amount > 0, "Amount must be positive"  # Stripped by -O flag!
    return amount * 1.15  # TODO: apply discount logic here

# ── TLS verification disabled (Security: TLSVerificationDisabled, OWASP A02) ─
import requests

def fetch_data(url):
    return requests.get(url, verify=False)  # MITM attack possible!

# ── Magic numbers (Bug: MagicNumber) ─────────────────────────────────────────
def apply_discount(price):
    if price > 5000:          # Magic: what is 5000?
        return price * 0.85   # Magic: what is 0.85?
    elif price > 1500:        # Magic: what is 1500?
        return price * 0.90   # Magic: what is 0.90?
    return price

# ── Unvalidated user input (Security: UnvalidatedInput, OWASP A01) ────────────
from flask import Flask, request as flask_request
app = Flask(__name__)

@app.route("/profile")
def profile():
    user_id = flask_request.args["user_id"]  # No validation!
    return f"<h1>User {user_id}</h1>"

if __name__ == "__main__":
    print("Running sample...")
    print(add_user_to_list("Alice"))
    print(hash_password("mypassword"))
    print(generate_token())
    print(apply_discount(6000))
