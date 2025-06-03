# backend/api/views.py
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Class, Attendance, Student
from .serializers import ClassSerializer, AttendanceSerializer, StudentSerializer

class ClassViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Provides list and retrieve for Class with nested students.
    """
    queryset = Class.objects.all()
    serializer_class = ClassSerializer

class AttendanceViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Provides list of Attendance filtered by class_id, week, and session.
    """
    queryset = Attendance.objects.all()
    serializer_class = AttendanceSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        class_id = self.request.query_params.get('class_id')
        week = self.request.query_params.get('week')
        session = self.request.query_params.get('session')
        if class_id is not None:
            qs = qs.filter(class_room_id=class_id)
        if week is not None:
            qs = qs.filter(week=week)
        if session is not None:
            qs = qs.filter(session=session)
        return qs

class StudentViewSet(viewsets.ModelViewSet):
    """
    Provides CRUD and login filtering for Student.
    """
    queryset = Student.objects.all()
    serializer_class = StudentSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        email = self.request.query_params.get('email')
        if email is not None:
            qs = qs.filter(email=email)
        return qs
