// src/components/Register.tsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

interface ClassItem {
  id: number;
  name: string;
}

const Register: React.FC = () => {
  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [selectedClassId, setSelectedClassId] = useState<string>('');
  const [name, setName] = useState('');
  const [studentId, setStudentId] = useState('');
  const [department, setDepartment] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
      fetch('/api/classes/')
      .then(res => res.json())
      .then((data: ClassItem[]) => {
        setClasses(data);
        if (data.length) setSelectedClassId(String(data[0].id));
      })
      .catch(console.error);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetch('/api/students/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, studentId, department, email, password, class_enrolled: selectedClassId })
    })
      .then(res => {
        if (!res.ok) throw new Error('Registration failed');
        alert('Registration successful!');
        navigate('/');
      })
      .catch(err => {
        console.error(err);
        alert('Registration failed');
      });
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-red-400 via-white to-blue-400">
      <h2 className="text-2xl font-bold mb-4">Student Registration</h2>
      <form onSubmit={handleSubmit} className="w-full max-w-sm bg-white shadow p-6 rounded-lg">
        <div className="mb-4">
          <label htmlFor="class" className="block text-gray-700 font-bold mb-2">Class:</label>
          <select
            id="class"
            value={selectedClassId}
            onChange={e => setSelectedClassId(e.target.value)}
            className="w-full border rounded px-3 py-2"
          >
            {classes.map(c => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </div>
        <div className="mb-4">
          <label htmlFor="name" className="block text-gray-700 font-bold mb-2">Name:</label>
          <input
            id="name"
            type="text"
            value={name}
            onChange={e => setName(e.target.value)}
            required
            className="w-full border rounded px-3 py-2"
          />
        </div>
        <div className="mb-4">
          <label htmlFor="studentId" className="block text-gray-700 font-bold mb-2">Student ID:</label>
          <input
            id="studentId"
            type="text"
            value={studentId}
            onChange={e => setStudentId(e.target.value)}
            required
            className="w-full border rounded px-3 py-2"
          />
        </div>
        <div className="mb-4">
          <label htmlFor="department" className="block text-gray-700 font-bold mb-2">Department:</label>
          <input
            id="department"
            type="text"
            value={department}
            onChange={e => setDepartment(e.target.value)}
            required
            className="w-full border rounded px-3 py-2"
          />
        </div>

        <div className="mb-4">
          <label htmlFor="email" className="block text-gray-700 font-bold mb-2">Email:</label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            required
            className="w-full border rounded px-3 py-2"
          />
        </div>
        
        <div className="mb-4">
          <label htmlFor="password" className="block text-gray-700 font-bold mb-2">Password:</label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
            className="w-full border rounded px-3 py-2"
          />
        </div>
        <div className="flex justify-between">
          <button
            type="button"
            onClick={() => navigate('/')}
            className="bg-gray-500 hover:bg-gray-700 text-white font-medium py-2 px-4 rounded-md"
          >
            Back to Login
          </button>
          <button
            type="submit"
            className="bg-green-500 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-md"
          >
            Submit
          </button>
        </div>
      </form>
    </div>
  );
};

export default Register;
