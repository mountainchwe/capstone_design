// src/components/Login.tsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { IoLogIn } from "react-icons/io5";
import type { Student } from '../types';

interface LoginProps {
  onLogin: (role: string) => void;
}

const Login: React.FC<LoginProps> = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    // Teacher mock credentials
    const teacherAccount = { email: 'seonyeong@khu.ac.kr', password: 'password' };
    if (email === teacherAccount.email && password === teacherAccount.password) {
      onLogin('teacher');
      navigate('/teacher');
      return;
    }

    // Fetch registered students fresh
    try {
      const res = await fetch(`/api/students/?email=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}`);
      const students: Student[] = await res.json();
      const registered = (students.length > 0 ? students[0] : null);
      if (registered) {
        localStorage.setItem('currentStudent', JSON.stringify(registered));
        onLogin('student');
        navigate('/student');
      } else {
        alert('Invalid credentials');
      }
    } catch (err) {
      console.error(err);
      alert('Login failed');
    }
  };

  const handleRegisterNav = () => {
    navigate('/register');
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-red-400 via-white to-blue-400">
      <form onSubmit={handleLogin} className="w-full max-w-sm bg-white shadow p-6 rounded-lg">
        <div className="flex justify-center mb-4">
          <IoLogIn className="text-5xl text-blue-500" />
        </div>
        <h2 className="text-2xl text-center font-bold mb-4">Login</h2>
        <p className="text-center text-gray-600 mb-4">Please enter your credentials</p>
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
            type="submit"
            className="bg-blue-500 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md"
          >
            Login
          </button>
          <button
            type="button"
            onClick={handleRegisterNav}
            className="bg-green-500 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-md"
          >
            Register
          </button>
        </div>
      </form>
    </div>
  );
};

export default Login;
