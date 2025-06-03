// src/components/TeacherDashboard.tsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import type { ClassData } from '../types';

const MAX_WEEKS = 16;

interface AttendanceRecord {
  student: {
    id: number;
    name: string;
    studentId: string;
    department: string;
    email: string;
  };
  status: string;
  timestamp: string;
}

interface TeacherDashboardProps {
  onLogout: () => void;
}

const TeacherDashboard: React.FC<TeacherDashboardProps> = ({ onLogout }) => {
  const [classes, setClasses] = useState<ClassData[]>([]);
  const [selectedClass, setSelectedClass] = useState<ClassData | null>(null);
  const [attendances, setAttendances] = useState<AttendanceRecord[]>([]);
  const [selectedTab, setSelectedTab] = useState(0);
  const [roster, setRoster] = useState<ClassData['students']>([]);
  const navigate = useNavigate();

  // Fetch classes
  useEffect(() => {
    fetch('/api/classes/')
      .then(res => res.json())
      .then((data: ClassData[]) => {
        setClasses(data);
        if (data.length) {
          setSelectedClass(data[0]);
          setRoster(data[0].students);
        }
      })
      .catch(console.error);

    // Refresh roster whenever selected class changes
  }, []);

  useEffect(() => {
    if (!selectedClass) return;
    fetch(`/api/classes/${selectedClass.id}/`)
      .then(res => res.json())
      .then((cls: ClassData) => {
        setRoster(cls.students);
      })
      .catch(console.error);
  }, [selectedClass]);

  // Fetch attendance when class or tab changes, and refresh roster
  useEffect(() => {
    if (!selectedClass) return;
    const week = Math.ceil((selectedTab + 1) / 2);
    const session = (selectedTab % 2) + 1;
    fetch(`/api/attendances/?class_id=${selectedClass.id}&week=${week}&session=${session}`)
      .then(res => res.json())
      .then((data: AttendanceRecord[]) => {
        setAttendances(data);
        // also refresh roster to include newly registered students
        fetch(`/api/classes/${selectedClass.id}/`)
          .then(res2 => res2.json())
          .then((cls: ClassData) => setRoster(cls.students))
          .catch(console.error);
      })
      .catch(console.error);
  }, [selectedClass, selectedTab]);

  const handleLogout = () => {
    onLogout();
    navigate('/');
  };

  if (!selectedClass) {
    return <div>Loading classes…</div>;
  }

  const tabLabels: string[] = [];
  for (let w = 1; w <= MAX_WEEKS; w++) {
    for (let s = 1; s <= 2; s++) {
      tabLabels.push(`Week ${w}-${s}`);
    }
  }

  // Merge roster and attendance records for display
  const displayRows = roster.map(student => {
    const rec = attendances.find(r => r.student.id === student.id);
    return {
      name: student.name,
      studentId: student.studentId,
      department: student.department,
      email: student.email,
      status: rec?.status ?? 'absent',
      timestamp: rec?.timestamp ?? '',
    };
  });

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 text-white border-r p-4">
        <h2 className="text-lg font-semibold mb-4 border-b-2">Classes</h2>
        <ul className="space-y-2">
          {classes.map(cls => (
            <li key={cls.id}>
              <button
                onClick={() => setSelectedClass(cls)}
                className={`w-full text-left px-4 py-2 rounded-md ${
                  selectedClass.id === cls.id ? 'bg-blue-500 text-white' : 'hover:bg-slate-600'
                }`}
              >
                {cls.name}
              </button>
            </li>
          ))}
        </ul>
      </aside>
      {/* Main Area */}
      <div className="flex-1 p-6 bg-slate-100 overflow-auto">
        <div className="flex justify-between mb-6">
          <button onClick={() => alert('Start face recognition attendance')}
            className="bg-green-500 hover:bg-green-700 text-white py-2 px-4 rounded-md"
            >
            Start Face Recognition
          </button>

          <button
            onClick={handleLogout}
            className="bg-red-500 hover:bg-red-700 text-white py-2 px-4 rounded-md"
          >
            Logout
          </button>
        </div>

        <h2 className="text-xl font-semibold mb-4">
          {selectedClass.name} – {tabLabels[selectedTab]} Attendance
        </h2>

        {/* Tabs for weeks and sessions */}
        <div className="flex overflow-x-auto space-x-2 mb-6">
          {tabLabels.map((label, idx) => (
            <button
              key={idx}
              onClick={() => setSelectedTab(idx)}
              className={`px-4 py-2 rounded-md border ${
                selectedTab === idx ? 'bg-blue-600 text-white' : 'bg-white hover:bg-blue-100'
              } whitespace-nowrap`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Attendance Records Table */}
        <table className="w-full bg-white shadow rounded-lg overflow-hidden">
          <thead className="bg-slate-900 ">
            <tr>
              {['Name','Student ID','Department','Email','Status','Time'].map((h, i) => (
                <th key={i} className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {displayRows.map((row, idx) => (
              <tr key={idx} className="hover:bg-gray-100">
                <td className="px-6 py-4">{row.name}</td>
                <td className="px-6 py-4">{row.studentId}</td>
                <td className="px-6 py-4">{row.department}</td>
                <td className="px-6 py-4">{row.email}</td>
                <td className="px-6 py-4">{row.status}</td>
                <td className="px-6 py-4">{row.timestamp || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default TeacherDashboard;
