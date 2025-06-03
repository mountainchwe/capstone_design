import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import type { Student } from '../types';

interface StudentDashboardProps {
  onLogout: () => void;
}

const StudentDashboard: React.FC<StudentDashboardProps> = ({ onLogout }) => {
  const [activeSection, setActiveSection] = useState<'profile' | 'face'>('profile');
  const [currentStudent, setCurrentStudent] = useState<Student | null>(null);
  const [showWebcam, setShowWebcam] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const stored = localStorage.getItem('currentStudent');
    if (stored) {
      setCurrentStudent(JSON.parse(stored));
    }
  }, []);

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside className="text-white w-64 bg-white dark:bg-red-800 border-r p-4">
        <h2 className="text-lg font-semibold text-white mb-4 border-b-2 border-white">
          Dashboard
        </h2>
        <ul className="mt-2 space-y-1">
          <li>
            <button
              onClick={() => setActiveSection('profile')}
              className={`w-full text-left px-4 py-2 rounded-md transition ${
                activeSection === 'profile'
                  ? 'bg-blue-500 text-white'
                  : 'hover:bg-gray-100 dark:hover:bg-slate-800'
              }`}
            >
              Student Profile
            </button>
          </li>
          <li>
            <button
              onClick={() => setActiveSection('face')}
              className={`w-full text-left px-4 py-2 rounded-md transition ${
                activeSection === 'face'
                  ? 'bg-blue-500 text-white'
                  : 'hover:bg-gray-100 dark:hover:bg-slate-800'
              }`}
            >
              Face Registration
            </button>
          </li>
        </ul>
      </aside>

      {/* Main Area */}
      <div className="flex-1 bg-slate-100 p-6 overflow-auto">
        {/* Logout */}
        <div className="flex justify-end mb-4">
          <button
            onClick={() => {
              localStorage.removeItem('currentStudent');
              onLogout();
              navigate('/');
            }}
            className="bg-blue-500 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md"
          >
            Logout
          </button>
        </div>

        {/* Conditional Sections */}
        {activeSection === 'profile' ? (
          // ── Student Profile Section ─
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="uppercase text-center text-2xl font-bold mb-4">Student’s Profile</h2>
            <div className="space-y-2 text-gray-800">
              <p><strong>Name:</strong> {currentStudent?.name}</p>
              <p><strong>Student ID:</strong> {currentStudent?.studentId}</p>
              <p><strong>Department:</strong> {currentStudent?.department}</p>
              <p><strong>Email:</strong> {currentStudent?.email}</p>
            </div>
          </div>
        ) : (
          // ── Face Registration Section ─
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="uppercase text-2xl font-bold mb-4">Face Vector Registration</h2>
            <p className="text-gray-600 mb-4">Click Start Webcam to preview, then capture when ready.</p>

            {!showWebcam && (
              <button
                onClick={async () => {
                  setShowWebcam(true);
                  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                  const video = document.getElementById("video") as HTMLVideoElement;
                  if (video) video.srcObject = stream;
                }}
                className="bg-blue-500 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md mb-4"
              >
                Start Webcam
              </button>
            )}

            {showWebcam && (
              <>
                <video id="video" width="320" height="240" autoPlay className="border mb-4" />
                <button
                  onClick={async () => {
                    const video = document.getElementById("video") as HTMLVideoElement;
                    const canvas = document.createElement("canvas");
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const context = canvas.getContext("2d");
                    if (!context) return;

                    context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const imageBase64 = canvas.toDataURL("image/jpeg");

                    const studentId = currentStudent?.studentId;
                    const name = currentStudent?.name;

                    if (!studentId || !name) {
                      alert("Missing student ID or name.");
                      return;
                    }

                    const res = await fetch("http://localhost:8001/register-face", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({
                        student_id: studentId,
                        name: name,
                        image_base64: imageBase64,
                      }),
                    });

                    const data = await res.json();
                    alert(data.message);
                  }}
                  className="bg-green-500 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-md"
                >
                  Capture & Register Face
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default StudentDashboard;