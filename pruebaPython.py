def count_items(items):
    counts = {}

    for item in items:
        counts[item] = counts.get(item, 0) + 1

    return counts



lista =["1","auto","casa","casa"]
total = count_items(lista)
print(total)
   