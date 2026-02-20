import { useCallback, useState } from "react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  onFile: (file: File) => void;
}

export function CsvDropzone({ onFile }: Props) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file?.name.endsWith(".csv")) onFile(file);
    },
    [onFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFile(file);
    },
    [onFile]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={cn(
        "border-2 border-dashed rounded-xl p-12 text-center transition-colors cursor-pointer",
        dragging
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/50"
      )}
    >
      <input
        type="file"
        accept=".csv"
        onChange={handleChange}
        className="hidden"
        id="csv-upload"
      />
      <label htmlFor="csv-upload" className="cursor-pointer space-y-3 block">
        <Upload className="mx-auto text-text-muted" size={40} />
        <p className="text-text-muted">
          Drag & drop a CSV file here, or{" "}
          <span className="text-primary underline">browse</span>
        </p>
        <p className="text-xs text-text-muted">
          Required columns: ticker, shares
        </p>
      </label>
    </div>
  );
}
