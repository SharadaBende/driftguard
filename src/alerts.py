import argparse
import os
import smtplib
from email.message import EmailMessage

import pandas as pd
import requests


def build_alert(row) -> tuple[str, str] | None:
    if row["overall"] != "red":
        return None
    reasons = []
    if row["input_level"] == "red":
        reasons.append(
            f"Input drift: {row['top_drift_feature']} PSI {row['max_input_psi']:.3f}"
        )
    if row["target_level"] == "red":
        reasons.append(f"Target (fare) drift: PSI {row['target_psi']:.3f}")
    if row["perf_level"] == "red":
        reasons.append(f"Performance: MAE is {row['mae_ratio']:.2f}x baseline")
    subject = f"[DriftGuard] CRITICAL drift detected in {row['month']}"
    body = subject + "\n\n" + "\n".join(f"- {r}" for r in reasons)
    body += "\n\nSuggested action: run src/retrain.py and compare champion vs challenger."
    return subject, body


def send_slack(text: str) -> bool:
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return False
    requests.post(url, json={"text": text}, timeout=15).raise_for_status()
    return True


def send_email(subject: str, body: str) -> bool:
    host, user = os.getenv("SMTP_HOST"), os.getenv("SMTP_USER")
    password, to = os.getenv("SMTP_PASSWORD"), os.getenv("ALERT_TO")
    if not all([host, user, password, to]):
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content(body)
    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587"))) as s:
        s.starttls()
        s.login(user, password)
        s.send_message(msg)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", help="e.g. 2023-10 (default: latest month)")
    args = parser.parse_args()

    summary = pd.read_csv("reports/drift_summary.csv")
    row = summary.iloc[-1] if not args.month else summary[summary["month"] == args.month].iloc[0]

    alert = build_alert(row)
    if alert is None:
        print(f"{row['month']}: status is {row['overall']}, no alert needed")
    else:
        subject, body = alert
        sent_slack = send_slack(body)
        sent_email = send_email(subject, body)
        if not (sent_slack or sent_email):
            print("DRY RUN (no Slack or email configured). Alert would be:\n")
            print(body)
        else:
            print(f"Alert sent (slack={sent_slack}, email={sent_email})")