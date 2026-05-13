import time


class PaymentService:
    """Handles payment processing and retry logic."""

    MAX_RETRIES = 3

    def process_payment(self, amount: float, card_token: str) -> dict:
        """Charge the card and return a transaction result."""
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self._charge(amount, card_token)
                return {"status": "success", "transaction_id": result}
            except PaymentGatewayError as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)

    def retry_failed_payment(self, transaction_id: str) -> dict:
        """Retry a previously failed payment."""
        return self._charge_by_id(transaction_id)

    def _charge(self, amount: float, token: str) -> str:
        # Simulate gateway call
        return "txn_abc123"

    def _charge_by_id(self, transaction_id: str) -> dict:
        return {"status": "retried", "transaction_id": transaction_id}


class PaymentGatewayError(Exception):
    pass
