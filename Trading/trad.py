from num2words import num2words
# principal = 10000  # Initial principal amount
# roi_schedule = [0.10, 0.10, 0.10, 0.15, -0.20]  # ROI schedule for each day
# days = 260  # Total number of days
acc_hp = 0

# for day in range(1, days + 1):
#     roi = roi_schedule[(day - 1) % len(roi_schedule)]  # Get the ROI for the current day
#     profit = principal * roi  # Calculate the profit for the current day
#     half_profit = profit / 2  # Calculate half of the profit
    
#     principal +=  half_profit  # Update the principal amount
#     acc_hp += half_profit
#     print(f"Day {day}: Pa = {principal}, P = {profit}, Hp = {half_profit}")
# total=principal+acc_hp
# print(total)
# Total= num2words(total)
# print(Total)


# principal = 10000  # Initial principal amount
# roi_schedule = [0.10, 0.10, 0.10, 0.15, -0.20]  # ROI schedule for each day
# days = 260  # Total number of days

# for day in range(1, days + 1):
#     roi = roi_schedule[(day - 1) % len(roi_schedule)]  # Get the ROI for the current day
#     profit = principal * roi  # Calculate the profit for the current day
#     half_profit1 = profit / 2  # Calculate half of the profit
#     half_profit2 = profit / 2
#     principal += half_profit1  # Add half of the profit to the principal amount
#     excess_amount = profit - half_profit2
#     if excess_amount > 2000:  # Check if the other half profit is greater than 2000
#         boom = excess_amount - 2000
#         principal += boom
#     acc_hp += excess_amount
#     print(f"Day {day}: Principal = {principal}, Profit = {profit}, Half Profit = {half_profit1}")
# print(excess_amount)
# print(boom)


principal = 10000  # Initial principal amount
roi_schedule = [0.05, 0.05, 0.05, 0.10, -0.0]  # ROI schedule for each day
days = 260  # Total number of days

for day in range(1, days + 1):
    roi = roi_schedule[(day - 1) % len(roi_schedule)]  # Get the ROI for the current day
    profit = principal * roi  # Calculate the profit for the current day
    half_profit1 = profit / 2
    half_profit2 = half_profit1
    principal += half_profit1   # Calculate half of the profit
    kharcha = 2000
    if half_profit2 > kharcha:  # Check if the other half profit is greater than 2000
        boom = half_profit2 - kharcha
        principal += boom
        acc_hp += kharcha
    
    print(f"Day {day}: Principal = {principal}, Profit = {profit}, Half Profit = {half_profit1}")
print(acc_hp)
print(principal)