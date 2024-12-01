# ex: creati o lista de nume de studenti si note. filtrati studentii cu note mai mari de 8

students = [("John", 7.2), ("Alice", 9.5), ("Bob", 8), ("Diana", 10), ("George", 6)]

def filterStudents(students):
    return [student for student in students if student[1] > 8]

print(filterStudents(students))