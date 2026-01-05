# notifications.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
import logging
from typing import List, Dict, Optional
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class EmailNotifier:
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        sender_email: str,
        admin_email: str
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender_email = sender_email
        self.admin_email = admin_email

    def send_trading_hours_alert(
        self,
        symbol: str,
        best_times: List[Dict],
        session_stats: Dict,
        image_path: Optional[str] = None
    ) -> bool:
        """Send an email with optimal trading times."""
        try:
            # Create message container
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.admin_email
            msg['Subject'] = f"Optimal Trading Times Alert - {symbol} - {datetime.utcnow().strftime('%Y-%m-%d')}"

            # Create HTML content
            html = f"""<html>
                <body>
                    <h2>Optimal Trading Times for {symbol}</h2>
                    <h3>Best Time Windows:</h3>
                    <ol>"""
            
            for i, window in enumerate(best_times[:3], 1):
                html += f"""
                    <li>{window['start']} - {window['end']} UTC
                        <ul>
                            <li>Win Rate: {window['score']*100:.1f}%</li>
                            <li>Total Trades: {window['total_trades']}</li>
                        </ul>
                    </li>"""
            
            html += "</ol><h3>Session Performance:</h3><ul>"
            
            for session, stats in session_stats.items():
                html += f"""
                    <li>{session}:
                        <ul>
                            <li>Win Rate: {stats['win_rate']*100:.1f}%</li>
                            <li>Total Trades: {stats['total_trades']}</li>
                            <li>Avg PnL: {stats['avg_pnl']:.2f}</li>
                        </ul>
                    </li>"""
            
            html += """</ul>
                <p>Happy Trading!<br>Your Trading Bot</p>
                </body>
            </html>"""

            # Attach HTML
            msg.attach(MIMEText(html, 'html'))

            # Attach performance image if available
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                    msg.attach(img)

            # Send email
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False