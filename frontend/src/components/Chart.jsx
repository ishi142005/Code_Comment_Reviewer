import { PieChart, Pie, Cell, Tooltip, Legend } from "recharts";

function Chart({ data }) {
  if (!data || !data.label_counts) return null;

  const chartData = Object.entries(data.label_counts).map(
    ([label, count]) => ({
      name: label,
      value: count
    })
  );

  const COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b"];

  return (
    <div style={{ margin: "20px 0" }}>
      <h3>Comment Distribution</h3>

      {chartData.length === 0 ? (
        <p>No data to display</p>
      ) : (
        <PieChart width={350} height={300}>
          <Pie
            data={chartData}
            dataKey="value"
            nameKey="name"
            outerRadius={100}
            label
          >
            {chartData.map((_, index) => (
              <Cell key={index} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip />
          <Legend />
        </PieChart>
      )}
    </div>
  );
}

export default Chart;