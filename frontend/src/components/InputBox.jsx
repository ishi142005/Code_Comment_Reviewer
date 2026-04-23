function InputBox({ prUrl, setPrUrl, onAnalyze, loading }) {
  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !loading) {
      onAnalyze();
    }
  };

  return (
    <div className="input-group">
      <input
        type="text"
        placeholder="Paste a GitHub PR URL to start analysis"
        value={prUrl}
        onChange={(e) => setPrUrl(e.target.value)}
        onKeyDown={handleKeyPress}
      />

      <button
        onClick={onAnalyze}
        disabled={loading || !prUrl.trim()}
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>
    </div>
  );
}

export default InputBox;