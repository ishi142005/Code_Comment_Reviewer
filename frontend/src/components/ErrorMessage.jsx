function ErrorMessage({ message }) {
    if (!message) return null;
  
    return (
      <div
        style={{
          marginTop: "10px",
          padding: "10px",
          background: "#fee2e2",
          color: "#991b1b",
          borderRadius: "8px"
        }}
      >
        {message}
      </div>
    );
  }
  
  export default ErrorMessage;