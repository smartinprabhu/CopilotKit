import { useEffect, useState } from 'react';

const CHUNK_SIZE = 500;

export const useChunkedChartData = (
  intradayForecast: any[],
  aggregationType: string,
  analysisType: string
) => {
  const [chartData, setChartData] = useState<any[]>([]);
  const [xAxisLabel, setXAxisLabel] = useState('');
  const [overallAverage, setOverallAverage] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (analysisType === 'dow' || !intradayForecast?.length) {
      setChartData([]);
      setXAxisLabel('');
      setOverallAverage(0);
      return;
    }

    let chunkIndex = 0;
    const chunks = Math.ceil(intradayForecast.length / CHUNK_SIZE);
    const grouped: { [key: string]: number } = {};
    setLoading(true);

    const processChunk = () => {
      const start = chunkIndex * CHUNK_SIZE;
      const end = start + CHUNK_SIZE;
      const chunk = intradayForecast.slice(start, end);

      for (const item of chunk) {
        let key = '';

        switch (aggregationType) {
          case 'halfhour':
            key = `${item.date}T${item.interval}`;
            break;

          case 'hourly':
            const hour = item.interval.split(':')[0];
            key = `${item.date}T${hour.padStart(2, '0')}:00`;
            break;

          case 'daily':
            key = item.date;
            break;

          case 'weekly': {
            const date = new Date(item.date);
            const weekStart = new Date(date);
            weekStart.setDate(date.getDate() - date.getDay());
            key = weekStart.toISOString().split('T')[0];
            break;
          }

          case 'monthly': {
            const date = new Date(item.date);
            key = `${date.getFullYear()}-${(date.getMonth() + 1)
              .toString()
              .padStart(2, '0')}`;
            break;
          }
        }

        if (!grouped[key]) grouped[key] = 0;
        grouped[key] += item.value || 0;
      }

      chunkIndex++;

      if (chunkIndex < chunks) {
        setTimeout(processChunk, 0); // Yield control to the browser
      } else {
        // Done: transform grouped data into chartData
        const finalData = Object.entries(grouped).map(([key, value]) => {
          let label = key;

          switch (aggregationType) {
            case 'halfhour': {
              const [date, time] = key.split('T');
              const dateLabel = new Date(date).toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                weekday: 'short',
              });
              label = `${dateLabel} ${time}`;
              setXAxisLabel('Date & Time');
              break;
            }

            case 'hourly': {
              const [date, time] = key.split('T');
              const dateLabel = new Date(date).toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                weekday: 'short',
              });
              label = `${dateLabel} ${time}`;
              setXAxisLabel('Date & Time');
              break;
            }

            case 'daily': {
              const dateLabel = new Date(key).toLocaleDateString('en-US', {
                weekday: 'short',
                month: 'short',
                day: 'numeric',
                year: '2-digit',
              });
              label = dateLabel;
              setXAxisLabel('Date');
              break;
            }

            case 'weekly': {
              const startDate = new Date(key);
              label = `Week ${startDate.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
              })}`;
              setXAxisLabel('Week');
              break;
            }

            case 'monthly': {
              const [year, month] = key.split('-');
              const date = new Date(parseInt(year), parseInt(month) - 1);
              label = date.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
              });
              setXAxisLabel('Month');
              break;
            }
          }

          return { label, value, date: key };
        });

        finalData.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
        const totalVolume = finalData.reduce((sum, item) => sum + item.value, 0);
        setOverallAverage(totalVolume / finalData.length);
        setChartData(finalData);
        setLoading(false);
      }
    };

    processChunk();
  }, [intradayForecast, aggregationType, analysisType]);

  return { chartData, xAxisLabel, overallAverage, loading };
};
