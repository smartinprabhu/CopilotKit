import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';

const generateRandomData = () => {
  const dates = Array.from({ length: 12 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (12 - i) * 5); // every 5 days
    return date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' });
  });

  return dates.map((date) => {
    const average = Math.floor(Math.random() * 10) + 10; // 10–20
    const median = average - Math.floor(Math.random() * 4); // lower than avg
    return {
      date,
      averageHandleTime: average,
      medianHandleTime: median,
    };
  });
};

const data = generateRandomData();

// Calculate overall total average (for reference line)
const totalAverage = (
  data.reduce((sum, d) => sum + d.averageHandleTime, 0) / data.length
).toFixed(2);

const AverageHandlingTimeChart = () => (
  <ResponsiveContainer width="100%" height={400}>
    <LineChart data={data} margin={{ top: 40, right: 30, left: 20, bottom: 5 }}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="date" />
      <YAxis label={{ value: 'Handle (Minutes)', angle: -90, position: 'insideLeft' }} />
      <Tooltip />
      <Legend />
      <ReferenceLine
        y={totalAverage}
        label="Total Average"
        stroke="red"
        strokeDasharray="3 3"
        strokeWidth={1}
      />
      <Line type="monotone" dataKey="averageHandleTime" name="Average Handle Time" stroke="#00bcd4" strokeWidth={2} />
      <Line type="monotone" dataKey="medianHandleTime" name="Median Handle Time" stroke="#ff9800" strokeWidth={2} />
    </LineChart>
  </ResponsiveContainer>
);

export default AverageHandlingTimeChart;
