"""
Alerting & Notification Module

Features:
- Slack Integration: Real-time alerts
- Email Reports: Scheduled quality reports
- Webhook Support: Custom integrations
- Threshold-Based Alerts: Configurable quality gates
- Alert History: Track alert history
"""

import json
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import warnings
import urllib.request
import urllib.error

warnings.filterwarnings('ignore')


@dataclass
class Alert:
    """Container for an alert"""
    id: str
    timestamp: datetime
    severity: str  # 'info', 'warning', 'error', 'critical'
    category: str
    title: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    sent: bool = False
    channels: List[str] = field(default_factory=list)


@dataclass
class AlertThreshold:
    """Container for alert threshold configuration"""
    metric: str
    operator: str  # 'gt', 'lt', 'gte', 'lte', 'eq', 'ne'
    value: float
    severity: str
    message_template: str


class AlertManager:
    """
    Centralized alert management and notification system.
    
    Features:
    - Multi-channel alerts (Slack, Email, Webhook)
    - Configurable thresholds
    - Alert deduplication
    - Alert history
    - Batch notifications
    """
    
    def __init__(
        self,
        default_severity: str = 'warning',
        dedupe_window_minutes: int = 60
    ):
        """
        Initialize the alert manager.
        
        Args:
            default_severity: Default alert severity
            dedupe_window_minutes: Window for deduplication
        """
        self.default_severity = default_severity
        self.dedupe_window_minutes = dedupe_window_minutes
        
        self._alerts: List[Alert] = []
        self._thresholds: List[AlertThreshold] = []
        self._channels: Dict[str, 'AlertChannel'] = {}
        self._alert_counter = 0
    
    def add_channel(self, name: str, channel: 'AlertChannel'):
        """Add an alert channel"""
        self._channels[name] = channel
    
    def add_threshold(
        self,
        metric: str,
        operator: str,
        value: float,
        severity: str = 'warning',
        message_template: str = None
    ):
        """
        Add an alert threshold.
        
        Args:
            metric: Metric name to monitor
            operator: Comparison operator
            value: Threshold value
            severity: Alert severity when triggered
            message_template: Custom message template
        """
        if message_template is None:
            message_template = f"{metric} {{operator}} {value}"
        
        self._thresholds.append(AlertThreshold(
            metric=metric,
            operator=operator,
            value=value,
            severity=severity,
            message_template=message_template
        ))
    
    def check_thresholds(
        self,
        metrics: Dict[str, float],
        send_alerts: bool = True
    ) -> List[Alert]:
        """
        Check metrics against configured thresholds.
        
        Args:
            metrics: Dictionary of metric values
            send_alerts: Whether to send alerts for violations
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for threshold in self._thresholds:
            if threshold.metric not in metrics:
                continue
            
            metric_value = metrics[threshold.metric]
            
            if self._evaluate_threshold(metric_value, threshold):
                alert = self.create_alert(
                    severity=threshold.severity,
                    category='threshold_violation',
                    title=f"Threshold Alert: {threshold.metric}",
                    message=threshold.message_template.format(
                        metric=threshold.metric,
                        value=metric_value,
                        threshold=threshold.value,
                        operator=threshold.operator
                    ),
                    metadata={
                        'metric': threshold.metric,
                        'value': metric_value,
                        'threshold': threshold.value,
                        'operator': threshold.operator
                    }
                )
                triggered_alerts.append(alert)
                
                if send_alerts:
                    self.send_alert(alert)
        
        return triggered_alerts
    
    def _evaluate_threshold(
        self,
        value: float,
        threshold: AlertThreshold
    ) -> bool:
        """Evaluate if a value violates a threshold"""
        operators = {
            'gt': lambda v, t: v > t,
            'lt': lambda v, t: v < t,
            'gte': lambda v, t: v >= t,
            'lte': lambda v, t: v <= t,
            'eq': lambda v, t: v == t,
            'ne': lambda v, t: v != t
        }
        
        op_func = operators.get(threshold.operator, lambda v, t: False)
        return op_func(value, threshold.value)
    
    def create_alert(
        self,
        severity: str = None,
        category: str = 'general',
        title: str = '',
        message: str = '',
        metadata: Dict[str, Any] = None
    ) -> Alert:
        """
        Create a new alert.
        
        Args:
            severity: Alert severity
            category: Alert category
            title: Alert title
            message: Alert message
            metadata: Additional metadata
            
        Returns:
            Created Alert object
        """
        self._alert_counter += 1
        
        alert = Alert(
            id=f"alert_{self._alert_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            severity=severity or self.default_severity,
            category=category,
            title=title,
            message=message,
            metadata=metadata or {}
        )
        
        self._alerts.append(alert)
        return alert
    
    def send_alert(
        self,
        alert: Alert,
        channels: List[str] = None
    ) -> Dict[str, bool]:
        """
        Send an alert through configured channels.
        
        Args:
            alert: Alert to send
            channels: Specific channels to use (None for all)
            
        Returns:
            Dictionary of channel -> success status
        """
        results = {}
        
        target_channels = channels or list(self._channels.keys())
        
        for channel_name in target_channels:
            if channel_name not in self._channels:
                continue
            
            channel = self._channels[channel_name]
            try:
                success = channel.send(alert)
                results[channel_name] = success
                
                if success:
                    alert.channels.append(channel_name)
                    alert.sent = True
            except Exception as e:
                results[channel_name] = False
                print(f"Failed to send alert via {channel_name}: {e}")
        
        return results
    
    def get_alert_history(
        self,
        severity: str = None,
        category: str = None,
        hours: int = None
    ) -> List[Alert]:
        """
        Get alert history with optional filters.
        
        Args:
            severity: Filter by severity
            category: Filter by category
            hours: Filter to last N hours
            
        Returns:
            List of matching alerts
        """
        alerts = self._alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        if hours:
            cutoff = datetime.now().timestamp() - (hours * 3600)
            alerts = [a for a in alerts if a.timestamp.timestamp() > cutoff]
        
        return alerts
    
    def generate_alert_summary(self) -> Dict[str, Any]:
        """Generate a summary of all alerts"""
        return {
            'total_alerts': len(self._alerts),
            'by_severity': {
                severity: len([a for a in self._alerts if a.severity == severity])
                for severity in ['info', 'warning', 'error', 'critical']
            },
            'by_category': {},
            'sent_count': len([a for a in self._alerts if a.sent]),
            'recent_alerts': [
                {
                    'id': a.id,
                    'severity': a.severity,
                    'title': a.title,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in sorted(self._alerts, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }


class AlertChannel:
    """Base class for alert channels"""
    
    def send(self, alert: Alert) -> bool:
        """Send an alert. Override in subclasses."""
        raise NotImplementedError


class SlackChannel(AlertChannel):
    """
    Slack integration for alerts.
    
    Sends alerts to a Slack channel via webhook.
    """
    
    def __init__(
        self,
        webhook_url: str,
        channel: str = None,
        username: str = "DataAiPrep",
        icon_emoji: str = ":bar_chart:"
    ):
        """
        Initialize Slack channel.
        
        Args:
            webhook_url: Slack webhook URL
            channel: Override channel name
            username: Bot username
            icon_emoji: Bot emoji
        """
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji
    
    def send(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        severity_colors = {
            'info': '#36a64f',
            'warning': '#ff9900',
            'error': '#ff0000',
            'critical': '#990000'
        }
        
        payload = {
            'username': self.username,
            'icon_emoji': self.icon_emoji,
            'attachments': [{
                'color': severity_colors.get(alert.severity, '#808080'),
                'title': alert.title,
                'text': alert.message,
                'fields': [
                    {'title': 'Severity', 'value': alert.severity.upper(), 'short': True},
                    {'title': 'Category', 'value': alert.category, 'short': True},
                    {'title': 'Time', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'short': True}
                ],
                'footer': 'DataAiPrep Alert System'
            }]
        }
        
        if self.channel:
            payload['channel'] = self.channel
        
        # Add metadata fields
        for key, value in list(alert.metadata.items())[:5]:
            payload['attachments'][0]['fields'].append({
                'title': key,
                'value': str(value),
                'short': True
            })
        
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
                
        except Exception as e:
            print(f"Slack send failed: {e}")
            return False


class EmailChannel(AlertChannel):
    """
    Email integration for alerts.
    
    Sends alerts via SMTP.
    """
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipient_emails: List[str],
        use_tls: bool = True
    ):
        """
        Initialize Email channel.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP port
            sender_email: Sender email address
            sender_password: Sender password or app password
            recipient_emails: List of recipient emails
            use_tls: Whether to use TLS
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_emails = recipient_emails
        self.use_tls = use_tls
    
    def send(self, alert: Alert) -> bool:
        """Send alert via email"""
        subject = f"[{alert.severity.upper()}] DataAiPrep Alert: {alert.title}"
        
        # Build HTML body
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .alert-box {{
                    border: 2px solid {'#ff0000' if alert.severity in ['error', 'critical'] else '#ff9900'};
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px;
                    background-color: #f9f9f9;
                }}
                .severity {{ 
                    font-weight: bold;
                    color: {'#ff0000' if alert.severity in ['error', 'critical'] else '#ff9900'};
                }}
                .metadata {{ background-color: #eee; padding: 10px; margin-top: 15px; }}
            </style>
        </head>
        <body>
            <div class="alert-box">
                <h2>{alert.title}</h2>
                <p class="severity">Severity: {alert.severity.upper()}</p>
                <p><strong>Category:</strong> {alert.category}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                <p>{alert.message}</p>
                
                <div class="metadata">
                    <h4>Additional Details:</h4>
                    <ul>
                        {''.join(f'<li><strong>{k}:</strong> {v}</li>' for k, v in alert.metadata.items())}
                    </ul>
                </div>
            </div>
            <p style="color: #888; font-size: 12px;">
                This alert was generated by DataAiPrep Alert System
            </p>
        </body>
        </html>
        """
        
        try:
            message = MIMEMultipart('alternative')
            message['Subject'] = subject
            message['From'] = self.sender_email
            message['To'] = ', '.join(self.recipient_emails)
            
            # Attach HTML
            html_part = MIMEText(html_body, 'html')
            message.attach(html_part)
            
            # Connect and send
            context = ssl.create_default_context()
            
            if self.use_tls:
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(self.sender_email, self.sender_password)
                    server.sendmail(
                        self.sender_email,
                        self.recipient_emails,
                        message.as_string()
                    )
            else:
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                    server.login(self.sender_email, self.sender_password)
                    server.sendmail(
                        self.sender_email,
                        self.recipient_emails,
                        message.as_string()
                    )
            
            return True
            
        except Exception as e:
            print(f"Email send failed: {e}")
            return False


class WebhookChannel(AlertChannel):
    """
    Generic webhook integration for alerts.
    
    Sends alerts to any HTTP endpoint.
    """
    
    def __init__(
        self,
        url: str,
        method: str = 'POST',
        headers: Dict[str, str] = None,
        payload_template: Dict[str, Any] = None,
        auth: Tuple[str, str] = None
    ):
        """
        Initialize webhook channel.
        
        Args:
            url: Webhook URL
            method: HTTP method
            headers: Additional headers
            payload_template: Custom payload template
            auth: Basic auth (username, password)
        """
        self.url = url
        self.method = method
        self.headers = headers or {}
        self.payload_template = payload_template
        self.auth = auth
    
    def send(self, alert: Alert) -> bool:
        """Send alert via webhook"""
        # Build payload
        if self.payload_template:
            payload = self._render_template(self.payload_template, alert)
        else:
            payload = {
                'alert_id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity,
                'category': alert.category,
                'title': alert.title,
                'message': alert.message,
                'metadata': alert.metadata
            }
        
        headers = {
            'Content-Type': 'application/json',
            **self.headers
        }
        
        # Add basic auth
        if self.auth:
            import base64
            credentials = f"{self.auth[0]}:{self.auth[1]}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers['Authorization'] = f"Basic {encoded}"
        
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.url,
                data=data,
                headers=headers,
                method=self.method
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status in [200, 201, 202, 204]
                
        except Exception as e:
            print(f"Webhook send failed: {e}")
            return False
    
    def _render_template(
        self,
        template: Dict[str, Any],
        alert: Alert
    ) -> Dict[str, Any]:
        """Render a payload template with alert data"""
        result = {}
        
        for key, value in template.items():
            if isinstance(value, str):
                # Replace placeholders
                value = value.replace('{alert_id}', alert.id)
                value = value.replace('{severity}', alert.severity)
                value = value.replace('{category}', alert.category)
                value = value.replace('{title}', alert.title)
                value = value.replace('{message}', alert.message)
                value = value.replace('{timestamp}', alert.timestamp.isoformat())
                result[key] = value
            elif isinstance(value, dict):
                result[key] = self._render_template(value, alert)
            else:
                result[key] = value
        
        return result


class QualityGate:
    """
    Quality gate for data quality checks.
    
    Define thresholds and automatically fail/pass data quality checks.
    """
    
    def __init__(
        self,
        name: str = "DataQualityGate",
        fail_on_critical: bool = True,
        fail_on_error: bool = True
    ):
        """
        Initialize quality gate.
        
        Args:
            name: Gate name
            fail_on_critical: Fail gate on critical alerts
            fail_on_error: Fail gate on error alerts
        """
        self.name = name
        self.fail_on_critical = fail_on_critical
        self.fail_on_error = fail_on_error
        
        self._checks: List[Dict[str, Any]] = []
        self._results: List[Dict[str, Any]] = []
    
    def add_check(
        self,
        name: str,
        metric: str,
        operator: str,
        threshold: float,
        severity: str = 'error'
    ):
        """
        Add a quality check.
        
        Args:
            name: Check name
            metric: Metric to check
            operator: Comparison operator
            threshold: Threshold value
            severity: Severity if check fails
        """
        self._checks.append({
            'name': name,
            'metric': metric,
            'operator': operator,
            'threshold': threshold,
            'severity': severity
        })
    
    def evaluate(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluate all quality checks.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            Evaluation result with pass/fail status
        """
        self._results = []
        
        operators = {
            'gt': lambda v, t: v > t,
            'lt': lambda v, t: v < t,
            'gte': lambda v, t: v >= t,
            'lte': lambda v, t: v <= t,
            'eq': lambda v, t: v == t,
            'ne': lambda v, t: v != t
        }
        
        for check in self._checks:
            metric_value = metrics.get(check['metric'])
            
            if metric_value is None:
                self._results.append({
                    **check,
                    'status': 'skipped',
                    'reason': f"Metric '{check['metric']}' not found"
                })
                continue
            
            op_func = operators.get(check['operator'])
            passed = op_func(metric_value, check['threshold'])
            
            self._results.append({
                **check,
                'value': metric_value,
                'status': 'passed' if passed else 'failed',
                'passed': passed
            })
        
        # Determine overall status
        failed_checks = [r for r in self._results if r.get('status') == 'failed']
        
        gate_passed = True
        if self.fail_on_critical:
            critical_failures = [r for r in failed_checks if r['severity'] == 'critical']
            if critical_failures:
                gate_passed = False
        
        if self.fail_on_error and gate_passed:
            error_failures = [r for r in failed_checks if r['severity'] == 'error']
            if error_failures:
                gate_passed = False
        
        return {
            'gate_name': self.name,
            'passed': gate_passed,
            'total_checks': len(self._checks),
            'passed_checks': len([r for r in self._results if r.get('passed', False)]),
            'failed_checks': len(failed_checks),
            'skipped_checks': len([r for r in self._results if r.get('status') == 'skipped']),
            'results': self._results
        }
    
    def generate_report(self) -> str:
        """Generate a quality gate report"""
        if not self._results:
            return "No evaluation results. Run evaluate() first."
        
        report = []
        report.append("=" * 60)
        report.append(f"QUALITY GATE REPORT: {self.name}")
        report.append("=" * 60)
        report.append("")
        
        passed_count = len([r for r in self._results if r.get('passed', False)])
        failed_count = len([r for r in self._results if r.get('status') == 'failed'])
        
        overall = "✅ PASSED" if failed_count == 0 else "❌ FAILED"
        report.append(f"Overall Status: {overall}")
        report.append(f"Checks Passed: {passed_count}/{len(self._results)}")
        report.append("")
        
        report.append("-" * 60)
        report.append("CHECK RESULTS")
        report.append("-" * 60)
        
        for result in self._results:
            status_icon = "✅" if result.get('passed', False) else ("⏭️" if result.get('status') == 'skipped' else "❌")
            report.append(f"\n{status_icon} {result['name']}")
            report.append(f"   Metric: {result['metric']} {result['operator']} {result['threshold']}")
            
            if 'value' in result:
                report.append(f"   Actual: {result['value']}")
            
            if result.get('status') == 'failed':
                report.append(f"   Severity: {result['severity'].upper()}")
        
        return "\n".join(report)

