"""
Tax education tools — reference data and illustrative calculators.
No API key required. All data is encoded as constants or computed with
standard financial formulas.

Note: Contribution limits and tax brackets are updated annually. These
reflect 2024/2025 figures; always verify with IRS.gov for the current year.
"""
from __future__ import annotations

import json

from langchain_core.tools import tool


# ── Reference data ────────────────────────────────────────────────────────────

_ACCOUNT_LIMITS_2025 = {
    "401k": {
        "employee_contribution": 23_500,
        "catch_up_50_plus": 7_500,
        "total_with_catch_up": 31_000,
        "employer_match_limit": 70_000,
        "notes": "Catch-up contribution of $7,500 for age 50+. Total employer+employee limit is $70,000.",
    },
    "traditional_ira": {
        "contribution_limit": 7_000,
        "catch_up_50_plus": 1_000,
        "total_with_catch_up": 8_000,
        "deductibility": "Deductible if you/spouse have no workplace plan; phased out at higher incomes if covered by workplace plan.",
        "income_phase_out_single": "$79,000–$89,000 (covered by workplace plan)",
        "income_phase_out_mfj": "$126,000–$146,000 (covered by workplace plan)",
    },
    "roth_ira": {
        "contribution_limit": 7_000,
        "catch_up_50_plus": 1_000,
        "total_with_catch_up": 8_000,
        "income_limit_single": "$150,000 (phase-out begins); $165,000 (ineligible)",
        "income_limit_mfj": "$236,000 (phase-out begins); $246,000 (ineligible)",
        "notes": "No RMDs during owner's lifetime. Qualified withdrawals are tax-free.",
    },
    "hsa": {
        "individual_limit": 4_300,
        "family_limit": 8_550,
        "catch_up_55_plus": 1_000,
        "eligibility": "Must be enrolled in a High-Deductible Health Plan (HDHP). Triple tax advantage: pre-tax contributions, tax-free growth, tax-free withdrawals for qualified medical expenses.",
    },
    "529": {
        "annual_gift_tax_exclusion": 18_000,
        "superfunding_5yr_lump_sum": 90_000,
        "notes": "No federal contribution limit; state limits vary ($300k–$550k). Superfunding allows 5 years of gift-tax exclusion upfront.",
    },
    "fsa": {
        "healthcare_fsa": 3_300,
        "dependent_care_fsa_single": 5_000,
        "dependent_care_fsa_mfj": 5_000,
        "notes": "Use-it-or-lose-it each year (employer may allow $640 rollover or 2.5-month grace period).",
    },
    "solo_401k": {
        "employee_contribution": 23_500,
        "employer_contribution_pct": "25% of net self-employment income",
        "total_limit": 70_000,
        "notes": "For self-employed individuals / sole proprietors with no full-time employees.",
    },
    "sep_ira": {
        "contribution_limit": "25% of net self-employment income or $70,000, whichever is less",
        "notes": "Employer-only contributions. Simple to set up for the self-employed.",
    },
}

_BRACKETS_2025 = {
    "single": [
        {"rate": 0.10, "up_to": 11_925},
        {"rate": 0.12, "up_to": 48_475},
        {"rate": 0.22, "up_to": 103_350},
        {"rate": 0.24, "up_to": 197_300},
        {"rate": 0.32, "up_to": 250_525},
        {"rate": 0.35, "up_to": 626_350},
        {"rate": 0.37, "up_to": None},
    ],
    "married_filing_jointly": [
        {"rate": 0.10, "up_to": 23_850},
        {"rate": 0.12, "up_to": 96_950},
        {"rate": 0.22, "up_to": 206_700},
        {"rate": 0.24, "up_to": 394_600},
        {"rate": 0.32, "up_to": 501_050},
        {"rate": 0.35, "up_to": 751_600},
        {"rate": 0.37, "up_to": None},
    ],
    "head_of_household": [
        {"rate": 0.10, "up_to": 17_000},
        {"rate": 0.12, "up_to": 64_850},
        {"rate": 0.22, "up_to": 103_350},
        {"rate": 0.24, "up_to": 197_300},
        {"rate": 0.32, "up_to": 250_500},
        {"rate": 0.35, "up_to": 626_350},
        {"rate": 0.37, "up_to": None},
    ],
}

_LTCG_BRACKETS_2025 = {
    "single":                {"0pct_up_to": 48_350,  "15pct_up_to": 533_400},
    "married_filing_jointly": {"0pct_up_to": 96_700,  "15pct_up_to": 600_050},
    "head_of_household":     {"0pct_up_to": 64_750,  "15pct_up_to": 566_700},
}


# ── tools ─────────────────────────────────────────────────────────────────────

@tool
def get_account_types_and_limits(account_type: str = "all") -> str:
    """
    Return contribution limits and key rules for US tax-advantaged accounts.

    account_type options: '401k', 'traditional_ira', 'roth_ira', 'hsa', '529',
    'fsa', 'solo_401k', 'sep_ira', or 'all' (returns all accounts).
    Data reflects 2025 IRS limits. Always verify current limits at IRS.gov.
    """
    try:
        key = account_type.lower().replace(" ", "_").replace("-", "_")
        if key == "all":
            data = _ACCOUNT_LIMITS_2025
        elif key in _ACCOUNT_LIMITS_2025:
            data = {key: _ACCOUNT_LIMITS_2025[key]}
        else:
            return json.dumps({
                "error": f"Unknown account type '{account_type}'.",
                "available": list(_ACCOUNT_LIMITS_2025.keys()),
            })
        return json.dumps({"tax_year": 2025, "accounts": data,
                           "source_note": "IRS 2025 limits. Verify at IRS.gov."})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_tax_brackets(filing_status: str = "single") -> str:
    """
    Return US federal income tax brackets and long-term capital gains rates.

    filing_status options: 'single', 'married_filing_jointly', 'head_of_household'.
    Returns ordinary income brackets and LTCG thresholds for 2025.
    """
    try:
        key = filing_status.lower().replace(" ", "_")
        if key not in _BRACKETS_2025:
            return json.dumps({
                "error": f"Unknown filing status '{filing_status}'.",
                "available": list(_BRACKETS_2025.keys()),
            })
        return json.dumps({
            "tax_year": 2025,
            "filing_status": filing_status,
            "ordinary_income_brackets": _BRACKETS_2025[key],
            "long_term_capital_gains": _LTCG_BRACKETS_2025.get(key, {}),
            "standard_deduction": {
                "single": 15_000,
                "married_filing_jointly": 30_000,
                "head_of_household": 22_500,
            }.get(key),
            "note": "Brackets apply to taxable income (after deductions). "
                    "NIIT of 3.8% applies to investment income above $200k single / $250k MFJ.",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def calculate_roth_vs_traditional(
    annual_contribution: float,
    years: int,
    current_marginal_tax_rate_pct: float,
    expected_retirement_tax_rate_pct: float,
    annual_return_pct: float = 7.0,
) -> str:
    """
    Compare Roth IRA vs Traditional IRA/401k on an after-tax basis.

    Provide: annual_contribution ($), years (investment horizon),
    current_marginal_tax_rate_pct (your tax rate today),
    expected_retirement_tax_rate_pct (expected tax rate in retirement),
    annual_return_pct (default 7%).

    Returns after-tax value at withdrawal for both account types and
    a recommendation based on the tax rate comparison.
    """
    try:
        r = annual_return_pct / 100
        current_rate = current_marginal_tax_rate_pct / 100
        retirement_rate = expected_retirement_tax_rate_pct / 100

        # Future value of annual contributions (end-of-year)
        if r > 0:
            fv = annual_contribution * ((1 + r) ** years - 1) / r * (1 + r)
        else:
            fv = annual_contribution * years

        # Traditional: contribute pre-tax, pay tax on withdrawal
        trad_after_tax = fv * (1 - retirement_rate)
        trad_tax_saved_now = annual_contribution * current_rate * years  # simplified

        # Roth: contribute post-tax (no deduction), withdraw tax-free
        roth_annual_contribution = annual_contribution * (1 - current_rate)  # after-tax dollars
        if r > 0:
            roth_fv = roth_annual_contribution * ((1 + r) ** years - 1) / r * (1 + r)
        else:
            roth_fv = roth_annual_contribution * years
        roth_after_tax = roth_fv  # no tax on withdrawal

        winner = "Traditional" if trad_after_tax > roth_after_tax else "Roth"
        margin = abs(trad_after_tax - roth_after_tax)

        if current_marginal_tax_rate_pct > expected_retirement_tax_rate_pct:
            reason = "Your tax rate is higher now than expected in retirement — Traditional saves more tax today."
        elif current_marginal_tax_rate_pct < expected_retirement_tax_rate_pct:
            reason = "Your tax rate is expected to be higher in retirement — Roth locks in today's lower rate."
        else:
            reason = "Tax rates are equal; Roth offers more flexibility (no RMDs, tax-free heirs)."

        return json.dumps({
            "annual_contribution": annual_contribution,
            "years": years,
            "annual_return_pct": annual_return_pct,
            "current_marginal_rate_pct": current_marginal_tax_rate_pct,
            "retirement_tax_rate_pct": expected_retirement_tax_rate_pct,
            "traditional": {
                "gross_value_at_retirement": round(fv, 2),
                "after_tax_value": round(trad_after_tax, 2),
                "upfront_tax_savings_over_period": round(trad_tax_saved_now, 2),
            },
            "roth": {
                "annual_after_tax_contribution": round(roth_annual_contribution, 2),
                "after_tax_value_at_retirement": round(roth_after_tax, 2),
                "note": "No RMDs; heirs inherit tax-free.",
            },
            "winner": winner,
            "advantage_amount": round(margin, 2),
            "reason": reason,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def calculate_effective_tax_rate(
    gross_income: float,
    filing_status: str = "single",
    pre_tax_deductions: float = 0.0,
    use_standard_deduction: bool = True,
) -> str:
    """
    Estimate federal effective and marginal tax rates for a given income.

    Provide: gross_income ($), filing_status ('single', 'married_filing_jointly',
    'head_of_household'), pre_tax_deductions ($ of 401k, HSA, etc.),
    use_standard_deduction (True/False).
    Returns marginal rate, effective rate, and a bracket breakdown.
    """
    try:
        key = filing_status.lower().replace(" ", "_")
        if key not in _BRACKETS_2025:
            return json.dumps({"error": f"Unknown filing status '{filing_status}'"})

        std_deductions = {"single": 15_000, "married_filing_jointly": 30_000,
                          "head_of_household": 22_500}
        agi = gross_income - pre_tax_deductions
        deduction = std_deductions.get(key, 15_000) if use_standard_deduction else 0
        taxable = max(0, agi - deduction)

        brackets = _BRACKETS_2025[key]
        tax_owed = 0.0
        marginal_rate = brackets[-1]["rate"]
        bracket_detail = []
        prev = 0

        for b in brackets:
            top = b["up_to"] or float("inf")
            if taxable <= prev:
                break
            taxable_in_bracket = min(taxable, top) - prev
            tax_in_bracket = taxable_in_bracket * b["rate"]
            tax_owed += tax_in_bracket
            if taxable_in_bracket > 0:
                bracket_detail.append({
                    "rate": f"{b['rate']*100:.0f}%",
                    "income_in_bracket": round(taxable_in_bracket, 2),
                    "tax_in_bracket": round(tax_in_bracket, 2),
                })
                marginal_rate = b["rate"]
            prev = top

        effective_rate = tax_owed / gross_income if gross_income else 0

        return json.dumps({
            "gross_income": gross_income,
            "pre_tax_deductions": pre_tax_deductions,
            "agi": round(agi, 2),
            "standard_deduction": deduction if use_standard_deduction else "not taken",
            "taxable_income": round(taxable, 2),
            "federal_tax_owed": round(tax_owed, 2),
            "marginal_rate_pct": round(marginal_rate * 100, 1),
            "effective_rate_pct": round(effective_rate * 100, 2),
            "bracket_breakdown": bracket_detail,
            "note": "Estimate only. Does not include FICA, state/local taxes, credits, or AMT.",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── exported collection ───────────────────────────────────────────────────────

TAX_EDUCATION_TOOLS = [
    get_account_types_and_limits,
    get_tax_brackets,
    calculate_roth_vs_traditional,
    calculate_effective_tax_rate,
]
