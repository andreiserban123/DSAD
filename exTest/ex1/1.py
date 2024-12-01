# Avem un array de prețuri ale unei acțiuni: [1, 6, 3, 6, 4, 9, 3, 4].
# Scrie un program care determină tranzacția (cumpărare-vânzare) cu cel mai mare profit posibil.


prices = [1, 6, 3, 6, 4, 9, 3, 4]


def maxProfit(prices):
    max_profit = 0
    for i in range(len(prices)):
        for j in range(i + 1, len(prices)):
            if prices[j] - prices[i] > max_profit:
                max_profit = prices[j] - prices[i]
    return max_profit


print(maxProfit(prices))

