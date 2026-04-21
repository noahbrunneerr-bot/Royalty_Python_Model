from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any


BaseType = Literal["revenue", "gross_profit", "net_cash", "custom"]
PaymentFrequency = Literal["annual", "quarterly", "monthly"]
CapType = Literal["none", "total_multiple", "absolute_amount"]
FloorType = Literal["none", "annual_minimum", "total_minimum"]
TerminationType = Literal["maturity", "cap_reached", "event"]


@dataclass
class CapRule:
    active: bool = False
    cap_type: CapType = "none"
    cap_value: Optional[float] = None

    def validate(self) -> None:
        if self.active:
            if self.cap_type == "none":
                raise ValueError("CapRule active=True but cap_type='none'.")
            if self.cap_value is None or self.cap_value <= 0:
                raise ValueError("CapRule requires positive cap_value when active=True.")


@dataclass
class FloorRule:
    active: bool = False
    floor_type: FloorType = "none"
    floor_value: Optional[float] = None

    def validate(self) -> None:
        if self.active:
            if self.floor_type == "none":
                raise ValueError("FloorRule active=True but floor_type='none'.")
            if self.floor_value is None or self.floor_value < 0:
                raise ValueError("FloorRule requires non-negative floor_value when active=True.")


@dataclass
class StepUpRule:
    threshold: float
    new_rate: float

    def validate(self) -> None:
        if self.threshold < 0:
            raise ValueError("StepUpRule threshold must be >= 0.")
        if not (0 <= self.new_rate <= 1):
            raise ValueError("StepUpRule new_rate must be between 0 and 1.")


@dataclass
class MilestoneRule:
    period: int
    amount: float
    trigger_type: str = "fixed"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.period < 0:
            raise ValueError("MilestoneRule period must be >= 0.")
        if self.amount < 0:
            raise ValueError("MilestoneRule amount must be >= 0.")


@dataclass
class CatchUpRule:
    active: bool = False
    carry_forward_shortfall: bool = True
    max_catch_up_periods: Optional[int] = None

    def validate(self) -> None:
        if self.max_catch_up_periods is not None and self.max_catch_up_periods <= 0:
            raise ValueError("max_catch_up_periods must be > 0 if provided.")


@dataclass
class TerminationRule:
    termination_type: TerminationType = "maturity"
    event_name: Optional[str] = None

    def validate(self) -> None:
        if self.termination_type == "event" and not self.event_name:
            raise ValueError("TerminationRule with termination_type='event' requires event_name.")


@dataclass
class RoyaltyContract:
    # Core terms
    contract_name: str
    base_type: BaseType
    royalty_rate: float
    start_period: int
    end_period: int
    payment_frequency: PaymentFrequency
    payment_lag_periods: int = 0

    # Protection / constraint terms
    cap_rule: CapRule = field(default_factory=CapRule)
    floor_rule: FloorRule = field(default_factory=FloorRule)
    termination_rule: TerminationRule = field(default_factory=TerminationRule)

    # Dynamic / optional terms
    step_up_rules: List[StepUpRule] = field(default_factory=list)
    milestones: List[MilestoneRule] = field(default_factory=list)
    catch_up_rule: CatchUpRule = field(default_factory=CatchUpRule)

    # PG3 mapping flags
    uses_pg3_waterfall: bool = False
    uses_net_cf_to_consortium: bool = False
    debt_linked_distribution: bool = False
    exit_value_linked: bool = False

    # Free metadata for documentation / thesis traceability
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.contract_name.strip():
            raise ValueError("contract_name must not be empty.")

        if self.base_type not in {"revenue", "gross_profit", "net_cash", "custom"}:
            raise ValueError(f"Invalid base_type: {self.base_type}")

        if not (0 <= self.royalty_rate <= 1):
            raise ValueError("royalty_rate must be between 0 and 1.")

        if self.start_period < 0:
            raise ValueError("start_period must be >= 0.")

        if self.end_period <= self.start_period:
            raise ValueError("end_period must be greater than start_period.")

        if self.payment_frequency not in {"annual", "quarterly", "monthly"}:
            raise ValueError(f"Invalid payment_frequency: {self.payment_frequency}")

        if self.payment_lag_periods < 0:
            raise ValueError("payment_lag_periods must be >= 0.")

        self.cap_rule.validate()
        self.floor_rule.validate()
        self.termination_rule.validate()
        self.catch_up_rule.validate()

        for rule in self.step_up_rules:
            rule.validate()

        for rule in self.milestones:
            rule.validate()

    @property
    def term_length(self) -> int:
        return self.end_period - self.start_period

    def get_applicable_rate(self, base_amount: float) -> float:
        """
        Returns the royalty rate applicable to a given base amount.
        Uses the highest triggered step-up rule if available.
        """
        applicable_rate = self.royalty_rate

        for rule in sorted(self.step_up_rules, key=lambda x: x.threshold):
            if base_amount >= rule.threshold:
                applicable_rate = rule.new_rate

        return applicable_rate

    def get_milestone_amount(self, period: int) -> float:
        """
        Sum of milestone payments for a given period.
        """
        return sum(m.amount for m in self.milestones if m.period == period)

    def as_dict(self) -> Dict[str, Any]:
        """
        Useful later for export / thesis documentation.
        """
        return {
            "contract_name": self.contract_name,
            "base_type": self.base_type,
            "royalty_rate": self.royalty_rate,
            "start_period": self.start_period,
            "end_period": self.end_period,
            "payment_frequency": self.payment_frequency,
            "payment_lag_periods": self.payment_lag_periods,
            "term_length": self.term_length,
            "uses_pg3_waterfall": self.uses_pg3_waterfall,
            "uses_net_cf_to_consortium": self.uses_net_cf_to_consortium,
            "debt_linked_distribution": self.debt_linked_distribution,
            "exit_value_linked": self.exit_value_linked,
            "metadata": self.metadata,
        }