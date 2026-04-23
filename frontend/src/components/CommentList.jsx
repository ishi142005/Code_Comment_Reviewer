function CommentList({ results }) {
  if (!results || results.length === 0) {
    return <p>No comments available.</p>;
  }

  return (
    <div className="comment-section">
      <h2>Comments</h2>

      {results.map((item, index) => (
        <div className="comment" key={index}>
          <div className="comment-label">{item.label}</div>
          <p>{item.comment}</p>
        </div>
      ))}
    </div>
  );
}

export default CommentList;