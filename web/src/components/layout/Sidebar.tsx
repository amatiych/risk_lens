import { NavLink } from "react-router-dom";
import { Upload, BarChart3, MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";
import { useApp } from "@/context/AppContext";

const links = [
  { to: "/", label: "Upload", icon: Upload },
  { to: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { to: "/chat", label: "Chat", icon: MessageSquare },
];

export function Sidebar() {
  const { analysis } = useApp();
  const hasAnalysis = !!analysis;

  return (
    <aside className="w-56 shrink-0 border-r border-border bg-surface h-screen sticky top-0 flex flex-col">
      <div className="p-4 border-b border-border">
        <h1 className="text-lg font-bold text-primary">Risk Lens</h1>
        <p className="text-xs text-text-muted">Portfolio Risk Analysis</p>
      </div>
      <nav className="flex-1 p-2 space-y-1">
        {links.map(({ to, label, icon: Icon }) => {
          const disabled = to !== "/" && !hasAnalysis;
          return (
            <NavLink
              key={to}
              to={disabled ? "#" : to}
              onClick={(e) => disabled && e.preventDefault()}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors",
                  isActive && !disabled
                    ? "bg-primary/10 text-primary"
                    : "text-text-muted hover:text-text hover:bg-surface-2",
                  disabled && "opacity-40 cursor-not-allowed"
                )
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          );
        })}
      </nav>
    </aside>
  );
}
