import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    
    pmf = math.exp(-lam) * (lam ** k) / math.factorial(k)

    cdf = sum(
        math.exp(-lam) * (lam ** i) / math.factorial(i)
        for i in range(k + 1)
    )

    return {
        "pmf": float(pmf),
        "cdf": float(cdf)
    }