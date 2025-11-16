import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

// Generate dummy monthly attrition rate data
const months = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

const data = months.map((month, index) => ({
  month,
  attritionRate: parseFloat((Math.random() * 2 + 1).toFixed(2)), // 1% to 3%
}));

const AttritionRateChart = () => (
  <ResponsiveContainer width="100%" height={400}>
    <LineChart data={data} margin={{ top: 40, right: 30, left: 20, bottom: 5 }}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="month" />
      <YAxis
        domain={[0, 4]}
        tickFormatter={(value) => `${value}%`}
        label={{ value: 'Attrition Rate', angle: -90, position: 'insideLeft' }}
      />
      <Tooltip formatter={(value) => `${value}%`} />
      <Legend />
      <Line
        type="monotone"
        dataKey="attritionRate"
        name="Attrition Rate"
        stroke="#fb8c00"
        strokeWidth={2}
        dot={{ r: 4, stroke: '#fb8c00', strokeWidth: 2, fill: '#fff' }}
        activeDot={{ r: 6 }}
      />
    </LineChart>
  </ResponsiveContainer>
);

export default AttritionRateChart;
