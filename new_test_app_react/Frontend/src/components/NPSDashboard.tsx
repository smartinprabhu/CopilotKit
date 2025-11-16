import React from 'react';
import {
  PieChart, Pie, Cell, Tooltip, Legend,
  ComposedChart, Bar, Line, XAxis, YAxis, ResponsiveContainer
} from 'recharts';
import GaugeChart from 'react-gauge-chart';

const kpiData = [
  { label: 'Total Shows', value: '2,000' },
  { label: 'Total Responses', value: '1,200' },
  { label: 'Response Rate', value: '3.5' },
  { label: 'Dismissed', value: '800' },
  { label: 'Qualification Responses', value: '480' },
];

const pieData = [
  { name: 'Promoters', value: 612 },
  { name: 'Passives', value: 101 },
  { name: 'Detractors', value: 101 },
];

const COLORS = ['#00C49F', '#FFBB28', '#FF4D4F'];

const npsHistory = [
  { name: 'Jan', promoters: 70, passives: 20, detractors: 10, score: 50 },
  { name: 'Feb', promoters: 60, passives: 25, detractors: 15, score: 40 },
  { name: 'Mar', promoters: 80, passives: 10, detractors: 10, score: 60 },
  { name: 'Apr', promoters: 65, passives: 20, detractors: 15, score: 45 },
];

export default function NpsDashboard() {
  return (
    <div className="p-6 space-y-6 min-h-screen transition-colors">
      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {kpiData.map((kpi) => (
          <div
            key={kpi.label}
            className="bg-white dark:bg-gray-700 rounded-lg shadow p-4 text-center"
          >
            <div className="text-sm text-gray-500 dark:text-gray-400">{kpi.label}</div>
            <div className="text-xl font-bold">{kpi.value}</div>
          </div>
        ))}
      </div>

      {/* Score + Pie */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-4 rounded-lg shadow">
          <h4 className="font-semibold mb-4">Score</h4>
          <GaugeChart
            id="nps-gauge"
            nrOfLevels={20}
            percent={0.53}
            textColor="currentColor"
            colors={['#FF4D4F', '#FFBB28', '#00C49F']}
          />
        </div>

        <div className="p-4 rounded-lg shadow">
          <h4 className="font-semibold mb-4">Responses</h4>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                cx="50%"
                cy="50%"
                outerRadius={80}
                label={({ name, value }) => `${name} (${value})`}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--tw-bg-opacity, 1)',
                  color: '#000',
                  borderRadius: 6,
                  border: '1px solid #ccc',
                }}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* NPS History */}
      <div className="p-4 rounded-lg shadow">
        <h4 className="font-semibold mb-4">NPS History</h4>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={npsHistory}>
            <XAxis dataKey="name" stroke="currentColor" />
            <YAxis stroke="currentColor" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#f9f9f9',
                color: '#000',
                border: '1px solid #ccc',
              }}
            />
            <Legend />
            <Bar dataKey="promoters" stackId="a" fill="#00C49F" />
            <Bar dataKey="passives" stackId="a" fill="#FFBB28" />
            <Bar dataKey="detractors" stackId="a" fill="#FF4D4F" />
            <Line type="monotone" dataKey="score" stroke="#8884d8" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
