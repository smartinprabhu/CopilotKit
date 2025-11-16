import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

// Sample data from your chart
const data = [
  { name: 'LESS THAN ONE YEAR', value: 15.3 },
  { name: 'ONE–TWO YEARS', value: 26 },
  { name: 'TWO–FIVE YEARS', value: 24.66 },
  { name: 'FIVE YEARS AND ABOVE', value: 34 },
];

const COLORS = ['#4f81bd', '#c0504d', '#9bbb59', '#8064a2'];

const TenurePieChart = () => (
  <ResponsiveContainer width="100%" height={400}>
    <PieChart>
      <Pie
        data={data}
        cx="50%"
        cy="50%"
        outerRadius={100}
        label={({ name, value }) => `${value}%`}
        dataKey="value"
      >
        {data.map((entry, index) => (
          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
        ))}
      </Pie>
                        <Tooltip
                            content={({ active, payload, label }) => {
                            if (!active || !payload?.length) return null;
                            
                            return (
                                <div className="p-2 rounded-lg border text-sm bg-white text-gray-900 border-gray-300 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">
                                
                                {payload.map(entry => (
                                    <div key={entry.name} className="flex items-center gap-2">
                                    <div style={{ width: 10, height: 10, backgroundColor: entry.color, borderRadius: 2 }}></div>
                                    <span>{entry.name}</span>
                                    <span className="font-bold">{`${entry.value}%`}</span>
                                    </div>
                                ))}
                                </div>
                            );
                            }}
                      />
      <Legend />
    </PieChart>
  </ResponsiveContainer>
);

export default TenurePieChart;
