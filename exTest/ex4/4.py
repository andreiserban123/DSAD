catalog = {
    "Serban": ("Focsani", 9.5),
    "Iulian": ("Arad", 7.5),
    "Maria": ("Focsani", 8.0),
    "Ana": ("Bucuresti", 10.0),
    "Vlad": ("Arad", 6.5),
    "Ioana": ("Bucuresti", 9.0),
    "Andrei": ("Cluj", 8.5),
    "Elena": ("Cluj", 9.0)
}


# grupare dupa oras

def groupByCity(catalog):
    cities = {}
    for student in catalog:
        city = catalog[student][0]
        if city not in cities:
            cities[city] = []
        cities[city].append(student)
    return cities


print(groupByCity(catalog))

# media notelor pe oras

def avgByCity(catalog):
    cities = groupByCity(catalog)
    avg = {}
    for city in cities:
        avg[city] = sum([catalog[student][1] for student in cities[city]]) / len(cities[city])
    return avg

avg = avgByCity(catalog)
print(avg)


# o noua structura care sa contina cheile  "Nume" -> lista de nume, "Oras" -> lista de orase, "Medie" -> lista de medii

def newStructure(catalog):
    cities = groupByCity(catalog)
    avg = avgByCity(catalog)
    newStruct = {"Nume": [], "Oras": [], "Medie": []}
    for city in cities:
        newStruct["Nume"].extend(cities[city])
        newStruct["Oras"].append(city)
        newStruct["Medie"].append(avg[city])
    return newStruct

print(newStructure(catalog))