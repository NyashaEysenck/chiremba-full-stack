import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Shield, Trash2 } from 'lucide-react';

// User type definition
export interface User {
  id: string;
  email: string;
  name: string;
  role: string;
  createdAt?: string;
}

// User Card Component props
interface UserCardProps {
  user: User;
  onDelete: (user: User) => void;
  onPromote: ((user: User) => void) | null;
}

const UserCard = ({ user, onDelete, onPromote }: UserCardProps) => {
  // Format date if available
  const formattedDate = user.createdAt 
    ? new Date(user.createdAt).toLocaleDateString() 
    : 'Unknown';
  
  // Get role badge color
  const getRoleBadgeColor = (role: string) => {
    switch (role) {
      case 'admin':
        return 'bg-purple-100 text-purple-800';
      case 'staff':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };
  
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex justify-between items-start">
          <div>
            <CardTitle>{user.name}</CardTitle>
            <CardDescription>{user.email}</CardDescription>
          </div>
          <div className={`px-2 py-1 rounded-full text-xs font-medium ${getRoleBadgeColor(user.role)}`}>
            {user.role}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pb-2">
        <p className="text-sm text-gray-500">
          Created: {formattedDate}
        </p>
      </CardContent>
      <CardFooter className="flex justify-end space-x-2">
        {onPromote && user.role !== 'admin' && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPromote(user)}
          >
            <Shield className="h-4 w-4 mr-2" />
            Make Admin
          </Button>
        )}
        <Button
          variant="destructive"
          size="sm"
          onClick={() => onDelete(user)}
        >
          <Trash2 className="h-4 w-4 mr-2" />
          Delete
        </Button>
      </CardFooter>
    </Card>
  );
};

export default UserCard;
