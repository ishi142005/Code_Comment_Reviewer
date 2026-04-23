function Summary({ data }) {
  if (!data || !data.label_counts) return null;

  return (
    <div className="summary">
      <h2>Total Comments: {data.total_comments}</h2>

      <div className="card-grid">
        {Object.entries(data.label_counts).map(([label, count]) => (
          <div className="stat-card" key={label}>
          <div className="label">{label}</div>
          <div className="value">{count}</div>
        </div>
        ))}
      </div>
    </div>
  );
}

export default Summary;