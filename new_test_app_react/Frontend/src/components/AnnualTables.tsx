import {
  Table,
  TableHeader,
  TableBody,
  TableHead,
  TableRow,
  TableCell,
} from "@/components/ui2/table";

const months = [
  "Jan-25", "Feb-25", "Mar-25", "Apr-25", "May-25", "Jun-25",
  "Jul-25", "Aug-25", "Sep-25", "Oct-25", "Nov-25", "Dec-25"
];

const annualMonthlyStrategic = [
  {
    label: "HC O/U Annual Lock Vs Monthly Strategic",
    values: [23, 13, 2, 5, 13, 30, -9, -5, 12, 21, 11, 17],
  },
  {
    label: "CPC Annual Planning",
    values: [22.91, 22.72, 19.75, 22.31, 22.28, 24.43, 24.68, 26.22, 23.70, 22.76, 23.34, 22.63],
  },
  {
    label: "CPC Monthly Strategic Planning",
    values: [27.85, 25.28, 20.08, 23.13, 24.42, 31.01, 23.22, 25.25, 25.84, 28.37, 26.09, 26.84],
  },
  {
    label: "CPC O/U",
    values: [4.94, 2.56, 0.33, 0.82, 2.14, 6.58, 1.46, 0.98, 2.14, 5.62, 2.75, 4.21],
  }
];

const premiumOrderSupportRatio = [
  {
    label: "Require Team Leaders HC",
    values: [6, 6, 7, 8, 7, 6, 8, 8, 7, 5, 5, 5],
  },
  {
    label: "Require Supervisor HC",
    values: [6, 6, 7, 8, 7, 6, 8, 8, 7, 5, 5, 5],
  }
];

const DataTable = ({ title, data }) => (
  <div className="mt-8">
    <Table>
      <TableHeader>
        <TableRow className="bg-gray-100 dark:bg-gray-800">
          <TableHead className="font-bold">Metric</TableHead>
          {months.map((month) => (
            <TableHead key={month} className="font-bold text-center">{month}</TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {data.map((row) => (
          <TableRow key={row.label}>
            <TableCell className="font-medium">{row.label}</TableCell>
            {row.values.map((val, idx) => (
              <TableCell key={idx} className="text-center">{val}</TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </div>
);

// Then in your page or component:
export default function StrategicTables() {
  return (
    <div className="p-4 space-y-10">
      <DataTable title="Annual Vs Monthly Strategic" data={annualMonthlyStrategic} />
      <DataTable title="Premium Order Support Ratio" data={premiumOrderSupportRatio} />
    </div>
  );
}
