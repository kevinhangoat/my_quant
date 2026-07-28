"""Alert notifiers for the momentum screener.

Each notifier implements ``send(title, message)``. Enable the ones you want via
the screener config. Credentials can come from the config file or environment
variables (env vars take precedence so secrets stay out of source control).

Phone notifications:
* ``TelegramNotifier`` - push to the Telegram mobile app via a bot.
* ``PushoverNotifier`` - push to the Pushover mobile app.
* ``WebhookNotifier``  - POST JSON to any custom endpoint (IFTTT, Zapier, etc.).
"""

import os
import smtplib
from email.message import EmailMessage
from typing import List, Optional

import requests


class Notifier:
    """Base notifier interface."""

    name = "base"

    def send(self, title: str, message: str) -> None:  # pragma: no cover
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    name = "console"

    def send(self, title: str, message: str) -> None:
        print(f"\n*** ALERT: {title} ***\n{message}\n")


class TelegramNotifier(Notifier):
    """Send a message through a Telegram bot to your phone.

    Setup: talk to @BotFather to create a bot and get ``bot_token``; send your
    bot a message, then read the chat id from
    ``https://api.telegram.org/bot<token>/getUpdates``.
    """

    name = "telegram"

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None,
                 timeout: float = 10.0) -> None:
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.timeout = timeout
        if not self.bot_token or not self.chat_id:
            raise ValueError("Telegram notifier needs bot_token and chat_id.")

    def send(self, title: str, message: str) -> None:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": f"{title}\n{message}"}
        requests.post(url, data=payload, timeout=self.timeout).raise_for_status()


class PushoverNotifier(Notifier):
    """Send a push notification via Pushover (https://pushover.net/)."""

    name = "pushover"

    def __init__(self, token: Optional[str] = None, user: Optional[str] = None,
                 timeout: float = 10.0) -> None:
        self.token = token or os.environ.get("PUSHOVER_TOKEN")
        self.user = user or os.environ.get("PUSHOVER_USER")
        self.timeout = timeout
        if not self.token or not self.user:
            raise ValueError("Pushover notifier needs token and user.")

    def send(self, title: str, message: str) -> None:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={"token": self.token, "user": self.user,
                  "title": title, "message": message},
            timeout=self.timeout,
        ).raise_for_status()


class EmailNotifier(Notifier):
    """Send alerts over SMTP (e.g. a Gmail app password).

    Configure via the screener config or environment variables. Credentials
    are optional at construction so the screener can run without email; if
    ``host``/``username``/``password`` are missing the notifier raises
    ``ValueError`` and ``build_notifiers`` skips it gracefully.

    Env vars: ``SMTP_HOST``, ``SMTP_PORT``, ``SMTP_USERNAME``,
    ``SMTP_PASSWORD``, ``EMAIL_FROM``, ``EMAIL_TO`` (comma-separated).
    """

    name = "email"

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        sender: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        *,
        use_tls: bool = True,
        timeout: float = 15.0,
    ) -> None:
        self.host = host or os.environ.get("SMTP_HOST")
        self.port = int(port or os.environ.get("SMTP_PORT", 587))
        self.username = username or os.environ.get("SMTP_USERNAME")
        self.password = password or os.environ.get("SMTP_PASSWORD")
        self.sender = sender or os.environ.get("EMAIL_FROM") or self.username
        env_to = os.environ.get("EMAIL_TO", "")
        self.recipients = recipients or [r.strip() for r in env_to.split(",") if r.strip()]
        self.use_tls = use_tls
        self.timeout = timeout
        if not (self.host and self.username and self.password):
            raise ValueError(
                "Email notifier needs host, username and password "
                "(set them in config or SMTP_* env vars)."
            )
        if not self.recipients:
            raise ValueError("Email notifier needs at least one recipient.")

    def send(self, title: str, message: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = title
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg.set_content(message)
        with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as server:
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)


class WebhookNotifier(Notifier):
    """POST the alert as JSON to a custom URL (IFTTT/Zapier/Discord/Slack...)."""

    name = "webhook"

    def __init__(self, url: Optional[str] = None, timeout: float = 10.0) -> None:
        self.url = url or os.environ.get("SCREENER_WEBHOOK_URL")
        self.timeout = timeout
        if not self.url:
            raise ValueError("Webhook notifier needs a url.")

    def send(self, title: str, message: str) -> None:
        requests.post(
            self.url, json={"title": title, "message": message}, timeout=self.timeout
        ).raise_for_status()


_NOTIFIER_TYPES = {
    ConsoleNotifier.name: ConsoleNotifier,
    TelegramNotifier.name: TelegramNotifier,
    PushoverNotifier.name: PushoverNotifier,
    EmailNotifier.name: EmailNotifier,
    WebhookNotifier.name: WebhookNotifier,
}


def build_notifiers(configs: List[dict]) -> List[Notifier]:
    """Instantiate notifiers from a list of ``{"type": ..., **kwargs}`` dicts.

    Disabled or misconfigured notifiers are skipped with a warning so a bad
    credential never stops the scanner from running.
    """
    notifiers: List[Notifier] = []
    for cfg in configs or []:
        cfg = dict(cfg)
        if not cfg.pop("enabled", True):
            continue
        ntype = cfg.pop("type", None)
        cls = _NOTIFIER_TYPES.get(ntype)
        if cls is None:
            print(f"[notifier] unknown type '{ntype}', skipping")
            continue
        try:
            notifiers.append(cls(**cfg))
        except (ValueError, TypeError) as exc:
            print(f"[notifier] skipping '{ntype}': {exc}")
    if not notifiers:
        notifiers.append(ConsoleNotifier())
    return notifiers
