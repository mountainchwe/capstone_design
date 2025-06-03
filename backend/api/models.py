from django.db import models

class Class(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Student(models.Model):
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=100)
    studentId = models.CharField(max_length=50)
    department = models.CharField(max_length=100)
    password = models.CharField(max_length=128, null=True, blank=True)
    class_enrolled = models.ForeignKey(Class, related_name='students', on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.name} ({self.studentId})"

class Attendance(models.Model):
    STATUS_PRESENT = 'present'
    STATUS_LATE = 'late'
    STATUS_ABSENT = 'absent'
    STATUS_CHOICES = [
        (STATUS_PRESENT, 'Present'),
        (STATUS_LATE, 'Late'),
        (STATUS_ABSENT, 'Absent'),
    ]

    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='attendances')
    class_room = models.ForeignKey(Class, on_delete=models.CASCADE, related_name='attendances')
    week = models.PositiveSmallIntegerField()
    session = models.PositiveSmallIntegerField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('student', 'class_room', 'week', 'session')

    def __str__(self):
        return f"{self.student.email} - Week {self.week}-{self.session}: {self.status}"
