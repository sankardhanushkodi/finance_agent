"""
Financial goal-planning tools — pure arithmetic, no API key required.

All tools use standard time-value-of-money formulas and return JSON strings
so they integrate cleanly with LangChain's tool-calling interface.
"""
from __future__ import annotations

import json
import math
from typing import Literal

from langchain_core.tools import tool


# ── helpers ──────────────────────────────────────────────────────────────────

def _monthly_rate(annual_return_pct: float) -> float:
    return annual_return_pct / 100 / 12


def _fv_lump_sum(pv: float, monthly_rate: float, months: int) -> float:
    """Future value of a lump-sum investment."""
    return pv * (1 + monthly_rate) ** months


def _fv_annuity(pmt: float, monthly_rate: float, months: int) -> float:
    """Future value of an ordinary annuity (end-of-period payments)."""
    if monthly_rate == 0:
        return pmt * months
    return pmt * ((1 + monthly_rate) ** months - 1) / monthly_rate


# ── tools ─────────────────────────────────────────────────────────────────────

@tool
def calculate_savings_needed(
    goal_amount: float,
    current_savings: float,
    years: float,
    annual_return_pct: float = 7.0,
) -> str:
    """
    Calculate the monthly savings required to reach a financial goal.

    Uses the future-value-of-annuity formula.
    Provide: goal_amount ($), current_savings ($), years (time horizon),
    annual_return_pct (expected annual return, default 7%).
    Returns monthly savings needed, total contributions, and total growth.
    """
    try:
        r = _monthly_rate(annual_return_pct)
        n = int(years * 12)

        # Current savings grow to this by the target date
        pv_grown = _fv_lump_sum(current_savings, r, n)
        remaining = goal_amount - pv_grown

        if remaining <= 0:
            return json.dumps({
                "goal_amount": goal_amount,
                "current_savings": current_savings,
                "years": years,
                "message": "Your current savings already exceed the goal at the given return rate.",
                "pv_grown_to": round(pv_grown, 2),
                "monthly_savings_needed": 0.0,
            })

        # Solve for monthly payment to make up the gap
        if r == 0:
            monthly = remaining / n
        else:
            monthly = remaining * r / ((1 + r) ** n - 1)

        total_contributions = current_savings + monthly * n
        total_growth = goal_amount - total_contributions

        return json.dumps({
            "goal_amount": goal_amount,
            "current_savings": current_savings,
            "years": years,
            "annual_return_pct": annual_return_pct,
            "monthly_savings_needed": round(monthly, 2),
            "total_contributions": round(total_contributions, 2),
            "total_growth_from_returns": round(total_growth, 2),
            "growth_pct_of_goal": round(total_growth / goal_amount * 100, 1),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def calculate_time_to_goal(
    goal_amount: float,
    current_savings: float,
    monthly_contribution: float,
    annual_return_pct: float = 7.0,
) -> str:
    """
    Calculate how long it will take to reach a savings goal.

    Provide: goal_amount ($), current_savings ($),
    monthly_contribution ($), annual_return_pct (default 7%).
    Returns years and months to goal, and a year-by-year milestone table.
    """
    try:
        r = _monthly_rate(annual_return_pct)
        balance = current_savings
        months = 0
        max_months = 600  # 50-year cap

        milestones = []
        while balance < goal_amount and months < max_months:
            balance = balance * (1 + r) + monthly_contribution
            months += 1
            if months % 12 == 0:
                milestones.append({
                    "year": months // 12,
                    "balance": round(balance, 2),
                })

        if months >= max_months and balance < goal_amount:
            return json.dumps({
                "error": "Goal cannot be reached within 50 years at this contribution rate. "
                         "Try increasing monthly contributions or the expected return."
            })

        years = months // 12
        remaining_months = months % 12

        return json.dumps({
            "goal_amount": goal_amount,
            "current_savings": current_savings,
            "monthly_contribution": monthly_contribution,
            "annual_return_pct": annual_return_pct,
            "years_to_goal": years,
            "months_to_goal": remaining_months,
            "total_months": months,
            "final_balance": round(balance, 2),
            "annual_milestones": milestones,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def calculate_retirement_number(
    desired_annual_income: float,
    current_age: int,
    retirement_age: int,
    current_savings: float = 0.0,
    monthly_contribution: float = 0.0,
    annual_return_pct: float = 7.0,
    inflation_rate_pct: float = 3.0,
    safe_withdrawal_rate_pct: float = 4.0,
) -> str:
    """
    Calculate how much you need to retire and whether you are on track.

    Uses the 4% safe withdrawal rule (configurable) and adjusts for inflation.
    Provide: desired_annual_income ($, in today's dollars), current_age,
    retirement_age, current_savings ($), monthly_contribution ($),
    annual_return_pct (default 7%), inflation_rate_pct (default 3%),
    safe_withdrawal_rate_pct (default 4%).
    Returns the nest egg needed, projected savings, and monthly savings gap.
    """
    try:
        years_to_retire = retirement_age - current_age
        if years_to_retire <= 0:
            return json.dumps({"error": "retirement_age must be greater than current_age"})

        # Inflate income to future dollars
        inflation_multiplier = (1 + inflation_rate_pct / 100) ** years_to_retire
        future_annual_income = desired_annual_income * inflation_multiplier

        # Nest egg needed (using safe withdrawal rate)
        nest_egg_needed = future_annual_income / (safe_withdrawal_rate_pct / 100)

        # Project current savings + contributions forward
        r = _monthly_rate(annual_return_pct)
        n = years_to_retire * 12
        projected = (
            _fv_lump_sum(current_savings, r, n)
            + _fv_annuity(monthly_contribution, r, n)
        )

        gap = nest_egg_needed - projected
        on_track = gap <= 0

        # Monthly savings needed to close the gap
        if gap > 0:
            if r == 0:
                additional_monthly = gap / n
            else:
                additional_monthly = gap * r / ((1 + r) ** n - 1)
        else:
            additional_monthly = 0.0

        return json.dumps({
            "current_age": current_age,
            "retirement_age": retirement_age,
            "years_to_retirement": years_to_retire,
            "desired_annual_income_today": desired_annual_income,
            "future_annual_income_inflated": round(future_annual_income, 2),
            "nest_egg_needed": round(nest_egg_needed, 2),
            "projected_savings_at_retirement": round(projected, 2),
            "savings_gap": round(gap, 2),
            "on_track": on_track,
            "additional_monthly_savings_needed": round(additional_monthly, 2),
            "assumptions": {
                "annual_return_pct": annual_return_pct,
                "inflation_rate_pct": inflation_rate_pct,
                "safe_withdrawal_rate_pct": safe_withdrawal_rate_pct,
            },
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def project_portfolio_growth(
    initial_amount: float,
    monthly_contribution: float,
    annual_return_pct: float,
    years: int,
    inflation_rate_pct: float = 0.0,
) -> str:
    """
    Project portfolio value year-by-year given an initial amount and monthly contributions.

    Provide: initial_amount ($), monthly_contribution ($),
    annual_return_pct (%), years (projection horizon),
    inflation_rate_pct (optional, adjusts to real dollars if > 0).
    Returns a list of {year, nominal_value, real_value} data points for charting.
    """
    try:
        r = _monthly_rate(annual_return_pct)
        real_deflator = (1 + inflation_rate_pct / 100)

        balance = initial_amount
        total_contributions = initial_amount
        data_points = [{"year": 0, "nominal_value": round(balance, 2),
                        "real_value": round(balance, 2),
                        "total_contributions": round(balance, 2)}]

        for year in range(1, years + 1):
            for _ in range(12):
                balance = balance * (1 + r) + monthly_contribution
                total_contributions += monthly_contribution

            real_value = balance / (real_deflator ** year) if inflation_rate_pct else balance
            data_points.append({
                "year": year,
                "nominal_value": round(balance, 2),
                "real_value": round(real_value, 2),
                "total_contributions": round(total_contributions, 2),
            })

        total_growth = balance - total_contributions
        return json.dumps({
            "initial_amount": initial_amount,
            "monthly_contribution": monthly_contribution,
            "annual_return_pct": annual_return_pct,
            "years": years,
            "final_value": round(balance, 2),
            "total_contributions": round(total_contributions, 2),
            "total_growth": round(total_growth, 2),
            "growth_multiplier": round(balance / initial_amount, 2) if initial_amount else None,
            "data_points": data_points,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def suggest_asset_allocation(
    years_to_goal: int,
    risk_tolerance: str,
    goal_type: str = "general",
) -> str:
    """
    Suggest an asset allocation (stocks/bonds/cash) based on time horizon,
    risk tolerance, and goal type.

    risk_tolerance options: 'conservative', 'moderate', 'aggressive'
    goal_type options: 'retirement', 'house', 'education', 'emergency', 'general'
    Returns recommended allocation percentages and rationale.
    """
    try:
        risk_tolerance = risk_tolerance.lower()
        goal_type = goal_type.lower()

        # Base stock % from time horizon (Rule of thumb: 110 - age proxy)
        if years_to_goal >= 20:
            base_stocks = 90
        elif years_to_goal >= 10:
            base_stocks = 75
        elif years_to_goal >= 5:
            base_stocks = 60
        elif years_to_goal >= 3:
            base_stocks = 40
        else:
            base_stocks = 20

        # Adjust for risk tolerance
        adjustments = {"conservative": -15, "moderate": 0, "aggressive": 10}
        stock_adj = adjustments.get(risk_tolerance, 0)
        stocks = max(5, min(95, base_stocks + stock_adj))

        # Emergency funds always stay in cash/short-term
        if goal_type == "emergency":
            return json.dumps({
                "allocation": {"cash_and_money_market": 100, "bonds": 0, "stocks": 0},
                "rationale": "Emergency funds must be fully liquid and principal-protected. "
                             "Use a high-yield savings account or money market fund.",
                "suggested_vehicles": ["High-yield savings account", "Money market fund", "Short-term T-bills"],
            })

        # Remainder split between bonds and cash
        bonds = max(0, min(80, 100 - stocks - 5))
        cash = 100 - stocks - bonds

        # Build vehicle suggestions
        stock_vehicles = []
        if years_to_goal >= 10:
            stock_vehicles = ["Total market index fund (VTI)", "S&P 500 ETF (VOO/SPY)",
                              "International index (VXUS)"]
        else:
            stock_vehicles = ["Dividend ETFs (VYM, SCHD)", "Large-cap value ETFs",
                              "Target-date fund"]

        bond_vehicles = ["Total bond market (BND)", "Treasury I-bonds (inflation-protected)",
                         "Short-term bond ETF (BSV)"] if bonds > 0 else []

        rationale_map = {
            "retirement": f"Long-term retirement goals benefit from growth-oriented allocation. "
                          f"With {years_to_goal} years, you have time to ride out market cycles.",
            "house": "House down-payment goals are medium-term. Balance growth with capital preservation "
                     "to protect against market downturns right before purchase.",
            "education": "Education savings benefit from a glide-path — start growth-oriented and "
                         "shift conservative as the enrollment date approaches.",
            "general": f"With {years_to_goal} years and {risk_tolerance} risk tolerance, "
                       f"this allocation balances growth potential against volatility.",
        }

        return json.dumps({
            "years_to_goal": years_to_goal,
            "risk_tolerance": risk_tolerance,
            "goal_type": goal_type,
            "allocation": {
                "stocks_pct": stocks,
                "bonds_pct": bonds,
                "cash_pct": cash,
            },
            "rationale": rationale_map.get(goal_type, rationale_map["general"]),
            "suggested_vehicles": {
                "stocks": stock_vehicles,
                "bonds": bond_vehicles,
                "cash": ["High-yield savings account", "Money market fund"],
            },
            "rebalancing_tip": "Review and rebalance annually, or if any asset class drifts >5% from target.",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def calculate_emergency_fund(
    monthly_expenses: float,
    months_coverage: int = 6,
    current_emergency_savings: float = 0.0,
) -> str:
    """
    Calculate emergency fund target and monthly savings needed to reach it.

    Provide: monthly_expenses ($), months_coverage (typically 3–6, default 6),
    current_emergency_savings ($ already saved, default 0).
    Returns target amount, gap, and savings plan.
    """
    try:
        target = monthly_expenses * months_coverage
        gap = max(0.0, target - current_emergency_savings)
        pct_complete = min(100.0, current_emergency_savings / target * 100) if target else 100.0

        # Savings schedule to fill gap in 6/12/18 months
        schedules = {}
        for fill_months in [6, 12, 18]:
            schedules[f"{fill_months}_months"] = round(gap / fill_months, 2) if gap else 0.0

        return json.dumps({
            "monthly_expenses": monthly_expenses,
            "months_coverage": months_coverage,
            "target_amount": round(target, 2),
            "current_savings": current_emergency_savings,
            "gap": round(gap, 2),
            "pct_complete": round(pct_complete, 1),
            "on_track": gap <= 0,
            "monthly_savings_to_fill_gap": schedules,
            "recommended_account": "High-yield savings account or money market fund — "
                                   "keep it accessible but separate from checking.",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── exported collection ───────────────────────────────────────────────────────

GOAL_TOOLS = [
    calculate_savings_needed,
    calculate_time_to_goal,
    calculate_retirement_number,
    project_portfolio_growth,
    suggest_asset_allocation,
    calculate_emergency_fund,
]
