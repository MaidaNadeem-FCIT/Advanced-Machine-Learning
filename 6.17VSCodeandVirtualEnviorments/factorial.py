def facto(num):
    fact = 1
    for i in range(1, num+1):
        fact = fact * i
    return fact

numb = 5 
rv = facto(numb)
print("factorial of {} is {}".format(numb, rv))