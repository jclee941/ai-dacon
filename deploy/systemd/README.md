# DACON auto-submit systemd units (user scope)

Persistent user timer that auto-submits the next-window top-5 candidates after the
DACON daily cap resets (KST 00:05), retrying on the daily-cap error.

## Install
```bash
cp deploy/systemd/dacon-auto-submit.* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now dacon-auto-submit.timer
loginctl enable-linger "$USER"   # survive logout
```

- `Persistent=true` -> catches up a missed run after reboot.
- Calls `scripts/auto_submit_next_window.sh qws941`.
- Log: `/tmp/dacon_auto_submit.log`.

## Status / disable
```bash
systemctl --user list-timers dacon-auto-submit.timer
systemctl --user disable --now dacon-auto-submit.timer
```
