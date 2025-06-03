# backend/api/admin.py
from django.contrib import admin
from .models import Class, Student, Attendance

@admin.register(Class)
class ClassAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ('email', 'name', 'studentId', 'department', 'class_enrolled')
    list_filter = ('department', 'class_enrolled')

@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ('student', 'class_room', 'week', 'session', 'status', 'timestamp')
    list_filter = ('week', 'session', 'status')
