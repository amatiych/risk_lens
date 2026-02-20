import type { ReactNode } from "react";

export function PageLayout({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="flex-1 overflow-auto">
      <header className="border-b border-border px-6 py-4">
        <h2 className="text-xl font-semibold">{title}</h2>
      </header>
      <main className="p-6">{children}</main>
    </div>
  );
}
